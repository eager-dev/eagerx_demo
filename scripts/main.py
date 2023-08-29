import eagerx
import eagerx_interbotix
import numpy as np
from datetime import datetime
import os


NAME = "HER_force_torque"
LOG_DIR = os.path.dirname(eagerx_interbotix.__file__) + f"/../logs/{NAME}_{datetime.today().strftime('%Y-%m-%d-%H%M')}"


if __name__ == "__main__":
    eagerx.set_log_level(eagerx.WARN)

    n_procs = 1
    rate = 10  # 20
    safe_rate = 20
    T_max = 10.0  # [sec]
    MUST_LOG = False
    MUST_TEST = False

    # Initialize empty graph
    graph = eagerx.Graph.create()

    # Create arm
    from eagerx_interbotix.xseries.xseries import Xseries

    robot_type = "vx300s"
    arm = Xseries.make(
        name=robot_type,
        robot_type=robot_type,
        sensors=["position", "velocity", "ee_pos", "ee_orn"],
        actuators=["pos_control", "gripper_control"],
        states=["position", "velocity", "gripper"],
        rate=rate,
    )
    arm.states.gripper.space.update(low=[0.0], high=[0.0])  # Set gripper to closed position
    arm.states.position.space.low[-2] = np.pi / 2
    arm.states.position.space.high[-2] = np.pi / 2
    graph.add(arm)

    # Connect actions
    graph.connect(action="position_control", target=arm.actuators.pos_control)
    graph.connect(action="gipper_control", target=arm.actuators.gripper_control)
    # Connecting observations
    graph.connect(source=arm.sensors.ee_pos, observation="ee_position")

    # Create camera
    from eagerx_interbotix.camera.objects import Camera

    cam = Camera.make(
        "cam",
        rate=rate,
        sensors=["image"],
        urdf=os.path.dirname(eagerx_interbotix.__file__) + "/camera/assets/realsense2_d435.urdf",
        optical_link="camera_color_optical_frame",
        calibration_link="camera_bottom_screw_frame",
        camera_index=0,  # todo: set correct index
    )
    graph.add(cam)
    # Create overlay
    from eagerx_interbotix.overlay.node import Overlay

    overlay = Overlay.make("overlay", rate=20, resolution=[480, 480], caption="robot view")
    graph.add(overlay)
    # Connect
    graph.connect(source=cam.sensors.image, target=overlay.inputs.main)
    graph.connect(source=cam.sensors.image, target=overlay.inputs.thumbnail)
    graph.render(source=overlay.outputs.image, rate=20, encoding="bgr")

    # Create backend
    from eagerx.backends.single_process import SingleProcess

    backend = SingleProcess.make()

    # Define engines
    from eagerx_pybullet.engine import PybulletEngine

    engine = PybulletEngine.make(rate=safe_rate, gui=True, egl=True, sync=True, real_time_factor=1)

    # Define environment
    from eagerx_demo.env import ArmEnv

    env = ArmEnv(
        name=f"cliport_demo",
        rate=rate,
        graph=graph,
        engine=engine,
        backend=backend,
        max_steps=int(T_max * rate),
        render_mode="human",
    )

    # Evaluate
    action_space = env.action_space
    for eps in range(5000):
        print(f"Episode {eps}")
        obs, info = env.reset()
        done = False
        while not done:
            action = action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
