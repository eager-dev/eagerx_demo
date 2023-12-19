from eagerx_demo.cliport.dataset import RavensDataset
from eagerx_demo.cliport import agents
from eagerx_demo.cliport.utils import utils

from pytorch_lightning import Trainer
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from eagerx_demo.utils import cam_spec_to_cam_config
from copy import deepcopy


def get_pick_place(act, ee_trans, ee_rot):
    r = R.from_quat(ee_rot).as_matrix()
    pick_pos = np.asarray(act["pose0"][0]) + ee_trans
    place_pos = np.asarray(act["pose1"][0]) + ee_trans
    pick_orn = np.asarray(act["pose0"][1])
    place_orn = np.asarray(act["pose1"][1])
    pick_orn = R.from_matrix(R.from_quat(pick_orn).as_matrix() @ r).as_quat()
    place_orn = R.from_matrix(R.from_quat(place_orn).as_matrix() @ r).as_quat()
    pick_place = dict(
        pick_pos=pick_pos.astype("float32"),
        place_pos=place_pos.astype("float32"),
        pick_orn=pick_orn.astype("float32"),
        place_orn=place_orn.astype("float32"),
    )
    return pick_place


def demonstration_pixels_to_act(act, points, bounds, pix_size):
    hmap = act["img"][:, :, 3]
    cart_points = []
    for point in points:
        cart_points.append(np.asarray(utils.pix_to_xyz(point, hmap, bounds, pix_size)))
    pick = [(points[0][0] + points[1][0]) // 2, (points[0][1] + points[1][1]) // 2]
    place = [(points[2][0] + points[3][0]) // 2, (points[2][1] + points[3][1]) // 2]
    p0_xyz = (cart_points[0] + cart_points[1]) / 2
    p1_xyz = (cart_points[2] + cart_points[3]) / 2
    p1_xyz[2] += p0_xyz[2]
    p0_theta = np.arctan2(cart_points[1][0] - cart_points[0][0], cart_points[1][1] - cart_points[0][1])
    p1_theta = np.arctan2(cart_points[3][0] - cart_points[2][0], cart_points[3][1] - cart_points[2][1])
    if p0_theta < -np.pi / 2:
        p0_theta += np.pi
    elif p0_theta > np.pi / 2:
        p0_theta -= np.pi
    if p1_theta < -np.pi / 2:
        p1_theta += np.pi
    elif p1_theta > np.pi / 2:
        p1_theta -= np.pi
    p0_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p0_theta))
    p1_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p1_theta))
    act["pose0"] = (np.asarray(p0_xyz), np.asarray(p0_xyzw))
    act["pose1"] = (np.asarray(p1_xyz), np.asarray(p1_xyzw))
    act["pick"] = [pick[0], pick[1], p0_theta]
    act["place"] = [place[0], place[1], p1_theta]
    return act

def act_to_demonstration_pixels(act, pix_size):
    # pixels = np.asarray(
    #     [
    #         [act["pick"][1], act["pick"][0]],
    #         [act["pick"][1], act["pick"][0]],
    #         [act["place"][1], act["place"][0]],
    #         [act["place"][1], act["place"][0]],
    #     ]
    # )
    
    # Gripper width is 0.08 m.
    # Take the gripper width and orientation into account when calculating the pick and place pixels.
    gripper_width = 0.02 / pix_size
    pick_theta = deepcopy(act["pick"][2])
    place_theta = deepcopy(act["place"][2])
    pixels = np.asarray(
        [
            [act["pick"][1] - np.sin(pick_theta) * gripper_width * 0.5, act["pick"][0] - np.cos(pick_theta) * gripper_width * 0.5],
            [act["pick"][1] + np.sin(pick_theta) * gripper_width * 0.5, act["pick"][0] + np.cos(pick_theta) * gripper_width * 0.5],
            [act["place"][1] - np.sin(place_theta) * gripper_width * 0.5, act["place"][0] - np.cos(place_theta) * gripper_width * 0.5],
            [act["place"][1] + np.sin(place_theta) * gripper_width * 0.5, act["place"][0] + np.cos(place_theta) * gripper_width * 0.5],
        ]
    )
    return pixels


def initialize_cliport(cfg):
    # Trainer
    trainer = Trainer(
        gpus=cfg["train"]["gpu"],
        fast_dev_run=cfg["debug"],
        checkpoint_callback=False,
        max_epochs=cfg["train"]["max_epochs"],
        max_steps=cfg["train"]["max_steps"],
        automatic_optimization=False,
        progress_bar_refresh_rate=0,
        weights_summary=None,
    )

    # Config
    data_dir = cfg["train"]["data_dir"]
    agent_type = cfg["train"]["agent"]
    n_demos = cfg["train"]["n_demos"]
    name = "{}-{}".format(agent_type, n_demos)

    # Create data_dir if it doesn't exist, plus action, color, depth, info and reward dirs.
    os.makedirs(data_dir, exist_ok=True)
    subdirs = ["action", "color", "depth", "info", "reward"]
    for subdir in subdirs:
        os.makedirs(os.path.join(data_dir, subdir), exist_ok=True)
        # Check how many demos are already in the data_dir.
        # TODO: Do we want to remove existing data or continue from where we left off?
        # Remove existing data in action, color, depth, info and reward dirs.
        # for f in os.listdir(os.path.join(data_dir, subdir)):
        #     os.remove(os.path.join(data_dir, subdir, f))

    demos = len(os.listdir(os.path.join(data_dir, subdir)))

    # Datasets
    cam_config = cam_spec_to_cam_config(cfg["cam_spec"])
    pix_size = cfg["pix_size"]
    in_shape = tuple(cfg["in_shape"])
    bounds = np.asarray(cfg["bounds"], dtype=np.float32).reshape(3, 2)
    train_ds = RavensDataset(
        data_dir, cfg, cam_config=cam_config, pix_size=pix_size, in_shape=in_shape, bounds=bounds, n_demos=demos, augment=True
    )
    train_ds.n_demos = demos
    train_ds.n_episodes = demos

    # Initialize agent
    agent = agents.names[agent_type](name, cfg, train_ds, None)
    if cfg["train"]["gpu"] is not None and len(cfg["train"]["gpu"]) > 0:
        device = "cuda"
        agent.to(device)
    else:
        device = "cpu"
    agent.eval()

    json_name = f"results-{name}.json"
    save_path = os.path.join(data_dir, "checkpoints")
    print(f"Save path for results: {save_path}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_json = os.path.join(save_path, f"{name}-{json_name}")
    checkpoint_path = os.path.join(save_path, f"{name}-last.ckpt")
    if os.path.exists(checkpoint_path) and cfg["train"]["load_from_last_ckpt"]:
        model_file = checkpoint_path
        print(f"Loading model from {model_file}")
        agent.load(model_file)
    return trainer, agent, train_ds, device, save_json, checkpoint_path, demos
