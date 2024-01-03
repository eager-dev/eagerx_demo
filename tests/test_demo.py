import eagerx
import eagerx_interbotix
import os
from eagerx_demo.utils import cam_config_to_cam_spec
from eagerx_demo.task import enginestates


def test_demo(engine="single_process"):
    prompt = ""
    colors = enginestates.COLORS
    for color_1 in colors.keys():
        for color_2 in colors.keys():
            prompt += f"Pick the {color_1} bolt and put it in the {color_2} nut. "

    robot_type = "panda"
    rtf = 0
    rate_env = 10
    rate_speech = 10
    rate_panda = 10
    rate_partnr = 10
    rate_engine = 20
    rate_cam = 20
    evaluate = False
    real = False
    ros = engine == "ros1"
    type_commands = True
    camera_window = 5

    # We assume we know the possible pick poses
    pick_poses = [
        [ 0.531, -0.208,  0.097],
        [ 0.572, -0.206,  0.097],
        [ 0.613, -0.205,  0.097],
        [ 0.652, -0.209,  0.097],
    ]
    pick_height = 0.097
    place_height = 0.11
    reset_pose = [.55, -0.075,  0.25,  1., 0.,  0.,  0]
    pix_size = 0.0015625
    bounds = [[pick_poses[0][0]-0.08, pick_poses[0][0]+0.17], [pick_poses[0][1] - 0.1, pick_poses[0][1]+0.4], [pick_height-0.13, place_height]]

    
    import numpy as np
    from eagerx_demo.realsense.cameras import RealSenseConfig

    cam_config = RealSenseConfig.CONFIG

    cam_config = [cam_config[0]] * camera_window

    image_size = cam_config[0]["image_size"]
    focal_len = cam_config[0]["intrinsics"][0]
    znear, zfar = cam_config[0]["zrange"]
    fovh = (image_size[0] / 2) / focal_len
    fovh = 180 * np.arctan(fovh) * 2 / np.pi

    cam_spec = cam_config_to_cam_spec(cam_config)

    # Initialize empty graph
    graph = eagerx.Graph.create()

    # Create arm
    from eagerx_franka.franka_arm.franka_arm import FrankaArm
    arm = FrankaArm.make(
        name=robot_type,
        robot_type=robot_type,
        sensors=["position", "ee_pos", "ee_orn", "gripper_position"],
        actuators=["moveit_to_ee_pose", "gripper_control"],
        states=["ee_pose", "velocity", "gripper"],
        rate=rate_panda,
        self_collision=False,
    )
    arm.config.sleep_positions = [0, 0, 0, -2.4, 0, 2.4, 0]
    arm.states.gripper.space.update(low=[0.0], high=[1.0])  # Set gripper to closed position
    graph.add(arm)

    # Create reset node
    from eagerx_demo.reset.node import ResetEEPose

    reset = ResetEEPose.make("reset", rate=rate_env, threshold=0.02, timeout=8.0)
    graph.add(reset)

    # Connect
    graph.connect(action="ee_pose", target=reset.feedthroughs.ee_pose)
    graph.connect(source=arm.states.ee_pose, target=reset.targets.goal_ee_pose)
    graph.connect(source=arm.states.gripper, target=reset.targets.goal_gripper)
    graph.connect(source=reset.outputs.ee_pose, target=arm.actuators.moveit_to_ee_pose)
    graph.connect(action="gripper", target=reset.feedthroughs.gripper)
    graph.connect(source=reset.outputs.gripper, target=arm.actuators.gripper_control)
    graph.connect(source=arm.sensors.ee_pos, target=reset.inputs.ee_pos)
    graph.connect(source=arm.sensors.ee_orn, target=reset.inputs.ee_orn)
    graph.connect(source=arm.sensors.ee_pos, observation="ee_pos")
    graph.connect(source=arm.sensors.ee_orn, observation="ee_orn")
    graph.connect(source=arm.sensors.position, observation="joint_pos")
    graph.connect(source=arm.sensors.gripper_position, observation="gripper_pos")

    # Create speech node
    from eagerx_demo.speech_recorder.objects import SpeechRecorder

    speech = SpeechRecorder.make(
        name="speech_recorder",
        rate=rate_speech,
        device="cuda",
        ckpt="tiny.en",
        prompt=prompt,
        type_commands=type_commands,
    )
    graph.add(speech)
    graph.connect(source=speech.sensors.speech, observation="speech")

    from eagerx_demo.partnr.node import Partnr
    from scipy.spatial.transform import Rotation as R

    ee_trans = [0, 0, 0]
    ee_rot = R.from_matrix([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).as_quat().tolist()

    partnr = Partnr.make(name="partnr", rate=rate_partnr, cam_spec=cam_spec, ee_trans=ee_trans, ee_rot=ee_rot, debug=True, evaluate=evaluate, pix_size=pix_size, bounds=bounds, camera_window=camera_window)
    graph.add(partnr)
    graph.connect(source=speech.sensors.speech, target=partnr.inputs.speech)
    graph.connect(source=partnr.outputs.pick_pos, observation="pick_pos")
    graph.connect(source=partnr.outputs.pick_orn, observation="pick_orn")
    graph.connect(source=partnr.outputs.place_pos, observation="place_pos")
    graph.connect(source=partnr.outputs.place_orn, observation="place_orn")

    # Create camera
    from eagerx_demo.realsense.objects import RealSense

    cam = RealSense.make(
        name="d435",
        rate=rate_cam,
        states=[],
        mode="rgbd",
        render_shape=list(image_size),
        base_pos=list(cam_config[0]["position"]),
        base_or=list(cam_config[0]["rotation"]),
        urdf=os.path.dirname(eagerx_interbotix.__file__) + "/camera/assets/realsense2_d435.urdf",
        optical_link="camera_bottom_screw_frame",
        calibration_link="camera_bottom_screw_frame",
        fov=float(fovh),
        near_val=float(znear),
        far_val=float(zfar),
    )
    graph.add(cam)

    # Connect
    graph.connect(source=cam.sensors.color, target=partnr.inputs.color, window=camera_window)
    graph.connect(source=cam.sensors.depth, target=partnr.inputs.depth, window=camera_window)

    # Create backend
    if ros: 
        from eagerx.backends.ros1 import Ros1
        backend = Ros1.make()
    else:
        from eagerx.backends.single_process import SingleProcess
        backend = SingleProcess.make()

    # Define engines
    if real:
        from eagerx_reality.engine import RealEngine
        engine = RealEngine.make(rate=rate_engine, sync=True)
    else:
        from eagerx_pybullet.engine import PybulletEngine
        engine = PybulletEngine.make(rate=rate_engine, gui=False, egl=False, sync=True, real_time_factor=rtf)

    # Add Dummy object 'task' with a single EngineState that creates a task (if the engine is a PybulletEngine)
    if engine.config.entity_id == "eagerx_pybullet.engine/PybulletEngine":
        # from eagerx_demo.task.enginestates import TaskState
        from eagerx_demo.task.agile import AgileTaskState
        from eagerx.core.space import Space
        task_es_name = "task"
        task_es = AgileTaskState.make(engine_pos=[pick_poses[0][0]+0.1, 0.0, -0.1], holder_pos=[pick_poses[0][0], pick_poses[0][1],  0.0], use_colors=["blue", "green", "yellow", "red"])
        engine.add_object(task_es_name, urdf=None)
        task_es_space = Space(low=0, high=1, shape=(), dtype="int64")  # var that specifies the task.
        engine._add_engine_state(task_es_name, "reset", task_es, task_es_space.to_dict())

        # Overwrite world_fn
        engine.config.world_fn = "eagerx_demo.task.agile/world_with_table_and_plane"

    # Define environment
    from eagerx_demo.env import ArmEnv

    env = ArmEnv(
        name=f"cliport_demo",
        robot_type=robot_type,
        rate=rate_env,
        graph=graph,
        engine=engine,
        backend=backend,
        render_mode=None,
        reset_ee_pose=reset_pose,  # Position of the arm when reset (out-of-view)
        reset_gripper=[1.0],  # Gripper position when reset (open)
        pick_poses=pick_poses,
        pick_height=pick_height,
        place_height=place_height,
        force_start=True,
    )


    # Evaluate
    action_space = env.action_space
    action = action_space.sample()
    for eps in range(2):
        print(f"Episode {eps}")
        obs, info = env.reset()
        done = False
        for step in range(3):
            obs, reward, terminated, truncated, info = env.step(obs)

if __name__ == "__main__":
    test_demo(engine="single_process")
    test_demo(engine="ros1")
