import eagerx
import eagerx_interbotix
import os
from eagerx_demo.utils import cam_config_to_cam_spec
from pynput.keyboard import Key, Listener
from moviepy.editor import ImageSequenceClip
from pathlib import Path
from eagerx_demo.cliport.dataset import augment_language


OBJECTS = ["bolt", "screw", "thing", "item", "fastener", "one"]
PICK_ACTIONS = ["pick", "grab", "take", "get"]
PLACE_ACTIONS = ["place", "put", "set", "drop"]
PREAMBLES = ["can you", "please", "will you", "could you", "would you", "can you please", "please can you", "please will you", "will you please", "could you please", "please could you", "could you please", "would you please", "please would you", "would you please", "can you please", "please can you", "please will you", "will you please", "could you please", "please could you", "could you please", "would you please", "please would you", "would you please"]
GREETINGS = ["hello", "hi", "hey", "howdy", "greetings", "good morning", "good afternoon", "good evening", "yo"]
GREETINGS = GREETINGS + [greeting + " robot" for greeting in GREETINGS]
TOP_LEFTS = ["top left", "upper left", "left top", "left upper"]
TOP_RIGHTS = ["top right", "upper right", "right top", "right upper"]
BOTTOM_LEFTS = ["bottom left", "lower left", "left bottom", "left lower", "lower left hand corner"]
BOTTOM_RIGHTS = ["bottom right", "lower right", "right bottom", "right lower", "lower right hand corner"]
MIDDLE_LEFTS = ["middle left", "left middle", "center left", "left center", "upper one of the lower left corner"]
MIDDLE_RIGHTS = ["middle right", "right middle", "center right", "right center", "upper one of the lower right corner"]
LEFT_TUBES = ["left tube", "left pipe", "left cylinder", "tube on the left", "pipe on the left", "cylinder on the left", "lefthand tube", "lefthand pipe", "lefthand cylinder", "tube on the lefthand side", "pipe on the lefthand side", "cylinder on the lefthand side", "tube on the left hand side", "pipe on the left hand side", "cylinder on the left hand side"]
RIGHT_TUBES = ["right tube", "right pipe", "right cylinder", "tube on the right", "pipe on the right", "cylinder on the right", "righthand tube", "righthand pipe", "righthand cylinder", "tube on the righthand side", "pipe on the righthand side", "cylinder on the righthand side", "tube on the right hand side", "pipe on the right hand side", "cylinder on the right hand side"]
MIDDLE_TUBES = ["middle tube", "middle pipe", "middle cylinder", "tube in the middle", "pipe in the middle", "cylinder in the middle"]
HOLES = ["hole", "opening", "insertion point", "cavity"]
ALL = OBJECTS + PICK_ACTIONS + PLACE_ACTIONS + PREAMBLES + GREETINGS + TOP_LEFTS + TOP_RIGHTS + BOTTOM_LEFTS + BOTTOM_RIGHTS + MIDDLE_LEFTS + MIDDLE_RIGHTS + LEFT_TUBES + RIGHT_TUBES + MIDDLE_TUBES + HOLES


def on_press(key):
    if key == Key.esc:
        # Stop listener
        global stop
        stop = True
        print("Stop")


if __name__ == "__main__":
    eagerx.set_log_level(eagerx.FATAL)
    prompt = ""
    colors = ["red", "blue", "green", "yellow"]
    locations = ["left tube", "right tube", "middle tube", "top left hole", "top right hole", "bottom left hole", "bottom right hole"]
    for _ in range(100):
        for color_1 in colors:
            for location in locations:
                prompt += augment_language(f"Pick the {color_1} bolt and put it in the {location}")

    robot_type = "panda"
    rtf = 0
    rate_env = 10
    rate_speech = 10
    rate_panda = 10
    rate_partnr = 10
    rate_engine = 20
    rate_cam = 20
    evaluate = False
    real = True
    ros = True
    type_commands = False
    camera_window = 5
    record_file = Path("record.mp4")
    stop = False

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

    partnr = Partnr.make(name="partnr", rate=rate_partnr, cam_spec=cam_spec, ee_trans=ee_trans, ee_rot=ee_rot, debug=False, evaluate=evaluate, pix_size=pix_size, bounds=bounds, camera_window=camera_window)
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
    graph.render(source=cam.sensors.color, rate=rate_cam, encoding="rgb")

    graph.gui()

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
        engine = PybulletEngine.make(rate=rate_engine, gui=True, egl=True, sync=True, real_time_factor=rtf)

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
        render_mode="human",
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
    # Collect events until released
    with Listener(on_press=on_press) as listener: 
        video_buffer = []    
        for eps in range(5000):
            if stop:
                break
            print(f"Episode {eps}")
            obs, info = env.reset()
            done = False
            stop = False
            while not done:
                obs, reward, terminated, truncated, info = env.step(obs)
                done = terminated or truncated or stop
                rgb = env.supervisor.get_last_image()
                if np.sum(rgb) > 0:
                    video_buffer.append(rgb)

    clip = ImageSequenceClip(video_buffer, fps=rate_cam)
    clip.write_videofile(str(record_file), fps=rate_cam)
    listener.join()
    env.shutdown()
