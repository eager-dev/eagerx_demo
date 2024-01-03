import eagerx
from eagerx import Space
from eagerx.utils.utils import Msg
from eagerx.core.specs import NodeSpec

import eagerx_demo
from eagerx_demo.partnr.demonstration import demonstrate
from eagerx_demo.partnr import utils
from eagerx_demo.utils import uint8_to_string

from threading import Thread
from pathlib import Path
import numpy as np
from time import sleep
from typing import List, Dict, Any


class Partnr(eagerx.Node):
    @classmethod
    def make(
        cls,
        name: str,
        rate: float,
        cam_spec: List[Dict[str, Any]],
        in_shape: List[int] = [320, 160, 6],
        pix_size: float = 0.0015625,
        bounds: List[float] = [[0.45, 0.7], [-0.3, 0.2], [-0.03, 0.5]],
        act_threshold=0.9,
        attn_temp=0.1,
        trans_temp=0.1,
        ee_trans=None,
        ee_rot=None,
        evaluate=False,
        camera_window=5,
        debug=False,
    ) -> NodeSpec:
        """
        PARTNR node.
        :param name: Node name
        :param rate: Rate at which callback is called.
        :return:
        """
        # Creates a base parameter specification object
        spec = cls.get_specification()
        root_dir = Path(eagerx_demo.__file__).parent.parent
        spec.config.update(
            name=name, rate=rate, process=eagerx.NEW_PROCESS
        )  # Update multiple parameters at once with .update(...)
        spec.config.inputs = ["color", "depth", "speech"]  # Or set individual parameters
        spec.config.outputs = ["pick_pos", "place_pos", "pick_orn", "place_orn"]
        spec.config.root_dir = str(root_dir)
        spec.config.tag = "default"
        spec.config.debug = debug
        spec.config.dataset = dict(images=True, cache=True, augment=dict(theta_sigma=60))
        spec.config.train = dict(
            exp_folder=str(root_dir / "exps"),
            train_dir=str(root_dir / "train_dir"),
            data_dir=str(root_dir / "data"),
            agent="dummy_cliport",
            n_demos=5,
            n_rotations=36,
            batchnorm=False,
            lr=1e-4,
            max_epochs=100,
            max_steps=1,
            attn_stream_fusion_type="add",
            trans_stream_fusion_type="conv",
            lang_fusion_type="mult",
            gpu=[0],
            n_val=0,
            save_steps=[],
            val_repeats=0,
            load_from_last_ckpt=True,
        )
        spec.config.cam_spec = cam_spec
        spec.config.pix_size = pix_size
        spec.config.in_shape = in_shape
        spec.config.bounds = bounds
        spec.config.act_threshold = act_threshold
        spec.config.attn_temp = attn_temp
        spec.config.trans_temp = trans_temp
        spec.config.ee_trans = ee_trans or [0, 0, 0]
        spec.config.ee_rot = ee_rot or [0, 0, 0, 1]
        spec.config.evaluate = evaluate
        spec.config.camera_window = camera_window
        return spec

    def initialize(self, spec: NodeSpec):
        self.act_threshold = spec.config.act_threshold
        self.attn_temp = spec.config.attn_temp
        self.trans_temp = spec.config.trans_temp
        self.ee_trans = spec.config.ee_trans
        self.ee_rot = spec.config.ee_rot
        self.debug = spec.config.debug
        self.evaluate = spec.config.evaluate
        self.pick_place = None
        self.updating = False
        self.inferring = False
        (
            self.trainer,
            self.agent,
            self.train_ds,
            self.device,
            self.save_json,
            self.checkpoint_path,
            episode,
        ) = utils.initialize_cliport(spec.config)
        self.episode_number = episode
        self.camera_window = spec.config.camera_window

    @eagerx.register.states()
    def reset(self):
        self.pick_place = None

    @eagerx.register.inputs(color=Space(dtype="uint8"), depth=Space(dtype="float32"), speech=Space(dtype="uint8"))
    @eagerx.register.outputs(
        pick_pos=Space(low=-1, high=1, shape=(3,), dtype="float32"),
        place_pos=Space(low=-1, high=1, shape=(3,), dtype="float32"),
        pick_orn=Space(low=0, high=1, shape=(4,), dtype="float32"),
        place_orn=Space(low=0, high=1, shape=(4,), dtype="float32"),
    )
    def callback(self, t_n: float, color: Msg, depth: Msg, speech: Msg):
        speech_data = speech.msgs[-1]
        color_data = color.msgs[-self.camera_window :]
        depth_data = depth.msgs[-self.camera_window :]
        pick_place = dict(
            pick_pos=np.zeros(3, dtype="float32"),
            place_pos=np.zeros(3, dtype="float32"),
            pick_orn=np.zeros(4, dtype="float32"),
            place_orn=np.zeros(4, dtype="float32"),
        )
        cmd = uint8_to_string(speech_data)

        # If command is not empty, run cliport, else update model
        if cmd != "":
            # Run cliport in separate thread
            print(f"Running cliport with command: {cmd}")
            self.inferring = True
            thread = Thread(target=self._act, args=(cmd, color_data, depth_data), daemon=True)
            thread.start()
        elif not self.evaluate and not self.updating and not self.inferring and self.episode_number > 0 and not self.debug:
            # Update model in separate thread
            self.updating = True
            thread = Thread(target=self._update_model, daemon=True)
            thread.start()

        # Return pick and place poses if they exist
        if self.pick_place is not None:
            pick_place = self.pick_place
            self.pick_place = None

        return pick_place

    def _act(self, cmd, color_data, depth_data):
        obs = {"color": (), "depth": ()}
        for data in color_data:
            obs["color"] += (data,)
        for data in depth_data:
            obs["depth"] += (data,)

        info = dict(lang_goal=cmd)

        # Wait for model update to finish
        while self.updating:
            self.backend.loginfo_once("Waiting for model update to finish.")
            sleep(0.1)

        self.agent.to(self.device)
        self.agent.eval()
        act = self.agent.act(obs, info, attn_temp=self.attn_temp, trans_temp=self.trans_temp)
        self.inferring = False
        confidence = act["pick_confidence"] * act["place_confidence"]
        print(f"PICK CONFIDENCE: {act['pick_confidence']}")
        print(f"PLACE CONFIDENCE: {act['place_confidence']}")
        print(act["pick"][:2])
        print(act["place"][:2])
        # Act if confidence is high enough, otherwise query for demonstration

        if not self.evaluate and confidence > self.act_threshold:
            self.pick_place = utils.get_pick_place(act, self.ee_trans, self.ee_rot)
        else:
            img = act["img"][:, :, :3]
            pixels = utils.act_to_demonstration_pixels(act, self.train_ds.pix_size)
            self.points = demonstrate(img, pixels)
            if np.sum(self.points) > 0:
                demo_act = utils.demonstration_pixels_to_act(act, self.points, self.train_ds.bounds, self.train_ds.pix_size)
                self.pick_place = utils.get_pick_place(demo_act, self.ee_trans, self.ee_rot)
                episode = (obs, demo_act, 1.0, info)
                self.train_ds.add(self.episode_number, [episode])
                self.episode_number += 1

    def _update_model(self):
        self.agent.train_ds = self.train_ds

        # Wait for inference to finish
        while self.inferring:
            sleep(0.1)

        self.trainer.fit(self.agent)
        self.trainer.current_epoch = 0
        self.trainer.global_step = 0
        temp_path = self.checkpoint_path.split(".ckpt")[0] + "_temp.ckpt"
        self.trainer.save_checkpoint(temp_path)
        # copy temp checkpoint to checkpoint path
        Path(temp_path).replace(self.checkpoint_path)
        self.updating = False
