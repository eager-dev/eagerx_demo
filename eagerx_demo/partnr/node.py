import eagerx
from eagerx import Space
from eagerx.utils.utils import Msg
from eagerx.core.specs import NodeSpec

import eagerx_demo
from eagerx_demo.utils import uint8_to_string
from eagerx_demo.cliport.dataset import RavensDataset
from eagerx_demo.cliport import agents
from eagerx_demo.cliport.utils import utils

from threading import Thread
import os
from pathlib import Path
from pytorch_lightning import Trainer
import numpy as np
import cv2
from copy import deepcopy
from time import sleep
import multiprocessing
from multiprocessing.sharedctypes import RawArray, Value
import ctypes


class Partnr(eagerx.Node):
    @classmethod
    def make(cls, name: str, rate: float, cam_config, act_threshold=0.9, attn_temp=0.001, trans_temp=1.0) -> NodeSpec:
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
        spec.config.debug = False
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
            load_from_last_ckpt=False,
        )
        spec.config.cam_config = cam_config
        spec.config.act_threshold = act_threshold
        spec.config.attn_temp = attn_temp
        spec.config.trans_temp = trans_temp

        return spec

    def initialize(self, spec: NodeSpec):
        self.act_threshold = spec.config.act_threshold
        self.attn_temp = spec.config.attn_temp
        self.trans_temp = spec.config.trans_temp
        self.pick_place = None
        self.updating = False
        self.inferring = False
        self._initialize_cliport(spec.config)
        self.episode_number = 0

    @eagerx.register.states()
    def reset(self):
        self.pick_place = None
        self.updating = False
        self.inferring = False

    @eagerx.register.inputs(color=Space(dtype="uint8"), depth=Space(dtype="float32"), speech=Space(dtype="uint8"))
    @eagerx.register.outputs(
        pick_pos=Space(low=-1, high=1, shape=(3,), dtype="float32"),
        place_pos=Space(low=-1, high=1, shape=(3,), dtype="float32"),
        pick_orn=Space(low=0, high=1, shape=(4,), dtype="float32"),
        place_orn=Space(low=0, high=1, shape=(4,), dtype="float32"),
    )
    def callback(self, t_n: float, color: Msg, depth: Msg, speech: Msg):
        speech_data = speech.msgs[-1]
        color_data = color.msgs[-1]
        depth_data = depth.msgs[-1]
        pick_place = dict(
            pick_pos=np.zeros(3, dtype="float32"),
            place_pos=np.zeros(3, dtype="float32"),
            pick_orn=np.zeros(4, dtype="float32"),
            place_orn=np.zeros(4, dtype="float32"),
        )
        cmd = uint8_to_string(speech_data)
        if cmd != "":
            # Run cliport in separate thread
            print(f"Running cliport with command: {cmd}")
            self.inferring = True
            thread = Thread(target=self._act, args=(cmd, color_data, depth_data), daemon=True)
            thread.start()
        elif not self.updating and not self.inferring and self.episode_number > 0:
            # Update model in separate thread
            self.updating = True
            thread = Thread(target=self._update_model, daemon=True)
            thread.start()
        if self.pick_place is not None:
            pick_place = self.pick_place
            self.pick_place = None
        return pick_place

    def _initialize_cliport(self, cfg):
        # Trainer
        self.trainer = Trainer(
            gpus=cfg["train"]["gpu"],
            fast_dev_run=cfg["debug"],
            checkpoint_callback=False,
            max_epochs=cfg["train"]["max_epochs"],
            max_steps=cfg["train"]["max_steps"],
            automatic_optimization=False,
        )

        # Config
        data_dir = cfg["train"]["data_dir"]
        agent_type = cfg["train"]["agent"]
        self.n_demos = cfg["train"]["n_demos"]
        name = "{}-{}".format(agent_type, self.n_demos)

        # Create data_dir if it doesn't exist, plus action, color, depth, info and reward dirs.
        os.makedirs(data_dir, exist_ok=True)
        subdirs = ["action", "color", "depth", "info", "reward"]
        for subdir in subdirs:
            os.makedirs(os.path.join(data_dir, subdir), exist_ok=True)
            # TODO: Do we want to remove existing data or continue from where we left off?
            # Remove existing data in action, color, depth, info and reward dirs.
            for f in os.listdir(os.path.join(data_dir, subdir)):
                os.remove(os.path.join(data_dir, subdir, f))

        # Datasets
        self.train_ds = RavensDataset(data_dir, cfg, n_demos=0, augment=True)
        self.train_ds.cam_config = cfg["cam_config"]

        # Initialize agent
        self.agent = agents.names[agent_type](name, cfg, self.train_ds, None)
        if len(cfg["train"]["gpu"]) > 0:
            self.device = "cuda"
            self.agent.to(self.device)
        self.agent.eval()

        # Save path for results.
        json_name = f"results-{name}.json"
        save_path = os.path.join(data_dir, "checkpoints")
        print(f"Save path for results: {save_path}")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.save_json = os.path.join(save_path, f"{name}-{json_name}")
        self.checkpoint_path = os.path.join(save_path, f"{name}-last.ckpt")

    def _act(self, cmd, color_data, depth_data):
        obs = {"color": (), "depth": ()}
        obs["color"] += (color_data,)
        obs["depth"] += (depth_data,)

        info = dict(lang_goal=cmd)

        while self.updating:
            print("Waiting for model update to finish...")
            sleep(0.1)
        print("Inferring...")
        act = self.agent.act(obs, info, attn_temp=self.attn_temp, trans_temp=self.trans_temp)
        print("Inference done.")
        confidence = act["pick_confidence"] * act["place_confidence"]
        print(confidence)
        if confidence > self.act_threshold:
            self.pick_place = dict(
                pick_pos=np.asarray(act["pose0"][0], dtype="float32"),
                place_pos=np.asarray(act["pose1"][0], dtype="float32"),
                pick_orn=np.asarray(act["pose0"][1], dtype="float32"),
                place_orn=np.asarray(act["pose1"][1], dtype="float32"),
            )
            self.pick_place = np.asarray(act["pose0"] + act["pose1"], dtype="float32").reshape(-1)
        else:
            shared_array_base = multiprocessing.Array(ctypes.c_int32, 4 * 2)
            shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
            shared_points = shared_array.reshape(4, 2)
            img = act["img"][:, :, :3]
            shape = img.shape
            shared_img = np.ndarray(shape, dtype="uint8", buffer=RawArray(ctypes.c_uint8, img.reshape(-1)))
            args = (shared_img, shape, shared_points)
            p = multiprocessing.Process(target=self._demonstrate, args=args)
            p.start()
            p.join()
            self.points = shared_points.tolist()

            if len(self.points) == 4:
                demo_act = self._demonstration_pixels_to_act(act, self.points, self.train_ds.bounds, self.train_ds.pix_size)
                episode = (obs, demo_act, 1.0, info)
                self.train_ds.add(self.episode_number, [episode])
                self.episode_number += 1
        self.inferring = False

    def _demonstration_pixels_to_act(self, act, points, bounds, pix_size):
        hmap = act["img"][:, :, 3]
        cart_points = []
        for point in points:
            cart_points.append(np.asarray(utils.pix_to_xyz(point, hmap, bounds, pix_size)))
        pick = [points[0][0] + points[1][0] // 2, points[0][1] + points[1][1] // 2]
        place = [points[2][0] + points[3][0] // 2, points[2][1] + points[3][1] // 2]
        p0_xyz = cart_points[0] + cart_points[1] / 2
        p1_xyz = cart_points[2] + cart_points[3] / 2
        p0_theta = np.arctan2(cart_points[1][1] - cart_points[0][1], cart_points[1][0] - cart_points[0][0])
        p1_theta = np.arctan2(cart_points[3][1] - cart_points[2][1], cart_points[3][0] - cart_points[2][0])
        p0_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p0_theta))
        p1_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p1_theta))
        act = {}
        act["pose0"] = (np.asarray(p0_xyz), np.asarray(p0_xyzw))
        act["pose1"] = (np.asarray(p1_xyz), np.asarray(p1_xyzw))
        act["pick"] = [pick[0], pick[1], p0_theta]
        act["place"] = [place[0], place[1], p1_theta]
        self.pick_place = dict(
            pick_pos=np.asarray(act["pose0"][0], dtype="float32"),
            place_pos=np.asarray(act["pose1"][0], dtype="float32"),
            pick_orn=np.asarray(act["pose0"][1], dtype="float32"),
            place_orn=np.asarray(act["pose1"][1], dtype="float32"),
        )
        return act

    @staticmethod
    def _demonstrate(img, shape, shared_points):
        print("Demonstrating...")
        img_original = img.view(dtype="uint8").reshape(shape)
        demo_window = DemonstrationWindow(img_original)
        demo_points = demo_window.demonstrate()
        print("Demonstration done.")
        for idx, point in enumerate(demo_points):
            shared_points[idx][0] = point[0]
            shared_points[idx][1] = point[1]

    def _update_model(self):
        self.agent.train_ds = self.train_ds
        while self.inferring:
            sleep(0.1)
        self.trainer.fit(self.agent)
        self.agent.to(self.device)
        self.agent.eval()
        self.updating = False
        self.trainer.current_epoch = 0
        self.trainer.global_step = 0
        self.trainer.save_checkpoint(self.checkpoint_path)


class DemonstrationWindow(object):
    def __init__(self, img):
        self.img_original = img
        self.img = deepcopy(img)
        self.img = np.asarray(self.img, dtype="uint8")
        self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
        self.points = []

    def demonstrate(self):
        window_name = "Demonstration"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self._mouse_callback)

        while True:
            cv2.imshow(window_name, self.img)
            k = cv2.waitKey(20) & 0xFF
            if k in [27, 10, 13]:
                break
            elif k == 8:
                if len(self.points) > 0:
                    self.img = deepcopy(self.img_original)
                    self.points.pop(-1)
                    for idx, point in enumerate(self.points):
                        cv2.circle(self.img, (point[0], point[1]), 2, (255, 0, 0), -1)
                        if idx in [1, 3]:
                            cv2.line(self.img, tuple(self.points[idx - 1]), tuple(self.points[idx]), (255, 255, 255), 2)
        cv2.destroyWindow(window_name)
        return self.points

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            cv2.circle(self.img, (x, y), 2, (255, 0, 0), -1)
            self.points.append([y, x])
            if len(self.points) in [2, 4]:
                cv2.line(
                    self.img,
                    tuple([self.points[-2][1], self.points[-2][0]]),
                    tuple([self.points[-1][1], self.points[-1][0]]),
                    (255, 255, 255),
                    2,
                )
