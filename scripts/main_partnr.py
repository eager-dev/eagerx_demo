import eagerx
import eagerx_interbotix
from datetime import datetime
import os


NAME = "OpenDR_demo"
LOG_DIR = os.path.dirname(eagerx_interbotix.__file__) + f"/../logs/{NAME}_{datetime.today().strftime('%Y-%m-%d-%H%M')}"


if __name__ == "__main__":
    # eagerx.set_log_level(eagerx.DEBUG)
    colors = ["blue", "cyan", "yellow", "gray", "brown", "orange", "pink", "purple"]

    prompt = ""
    for color1 in colors:
        for color2 in colors:
            prompt += f"Put the {color1} bolt in the {color2} nut.  "


    MUST_LOG = False
    robot_type = "vx300s"
    rtf = 0
    rate_env = 10
    rate_speech = 10
    rate_xseries = 10
    rate_partnr = 10
    rate_engine = 20
    rate_cam = 20

    # Initialize empty graph
    graph = eagerx.Graph.create()

    # Create arm
    from eagerx_interbotix.xseries.xseries import Xseries

    arm = Xseries.make(
        name=robot_type,
        robot_type=robot_type,
        sensors=["moveit_status", "position", "ee_pos", "ee_orn", "gripper_position"],
        actuators=["moveit_to", "gripper_control"],
        states=["position", "velocity", "gripper"],
        rate=rate_xseries,
    )
    arm.states.gripper.space.update(low=[0.0], high=[1.0])  # Set gripper to closed position
    arm.states.position.space.low[:] = arm.config.joint_lower
    arm.states.position.space.high[:] = arm.config.joint_upper
    graph.add(arm)

    # Create TaskSpaceControl
    from eagerx_demo.ik.node import TaskSpaceControl

    ik = TaskSpaceControl.make("task_space",
                               rate=rate_env,
                               joints=arm.config.joint_names,
                               upper=arm.config.joint_upper,
                               lower=arm.config.joint_lower,
                               ee_link=arm.config.gripper_link,
                               rest_poses=arm.config.sleep_positions,
                               gui=False,
                               robot_dict={"urdf": arm.config.urdf,
                                           "basePosition": arm.config.base_pos,
                                           "baseOrientation": arm.config.base_or})
    graph.add(ik)

    # Create reset node
    from eagerx_demo.reset.node import ResetArm

    reset = ResetArm.make("reset", rate=rate_env,
                          upper=arm.config.joint_upper,
                          lower=arm.config.joint_lower,
                          threshold=0.02, timeout=8.0)
    graph.add(reset)

    # Connect
    graph.connect(action="ee_pos", target=ik.inputs.ee_pos)
    graph.connect(action="ee_orn", target=ik.inputs.ee_orn)
    graph.connect(source=arm.sensors.position, target=ik.inputs.position)
    graph.connect(source=ik.outputs.goal, target=reset.feedthroughs.joints)
    graph.connect(source=arm.states.position, target=reset.targets.goal_joints)
    graph.connect(source=arm.states.gripper, target=reset.targets.goal_gripper)
    graph.connect(source=reset.outputs.joints, target=arm.actuators.moveit_to)
    graph.connect(action="gripper", target=reset.feedthroughs.gripper)
    graph.connect(source=reset.outputs.gripper, target=arm.actuators.gripper_control)
    graph.connect(source=arm.sensors.position, target=reset.inputs.joints)
    graph.connect(source=arm.sensors.ee_pos, observation="ee_pos")
    graph.connect(source=arm.sensors.ee_orn, observation="ee_orn")
    graph.connect(source=arm.sensors.moveit_status, observation="moveit_status", skip=True)
    graph.connect(source=arm.sensors.position, observation="joint_pos")
    graph.connect(source=arm.sensors.gripper_position, observation="gripper_pos")

    # Create speech node
    from eagerx_demo.speech_recorder.objects import SpeechRecorder

    speech = SpeechRecorder.make(
        name="speech_recorder",
        rate=rate_speech,
        debug=True,
        device="cpu",
        ckpt="base.en",
        prompt=prompt,
    )
    graph.add(speech)
    graph.connect(source=speech.sensors.speech, observation="speech")

    from eagerx_demo.partnr.node import Partnr
    from eagerx_demo.cliport.tasks.cameras import RealSenseD415
    from scipy.spatial.transform import Rotation as R

    cam_config = RealSenseD415
    cam_config = cam_config.CONFIG[0]
    cam_config["image_size"] = list(cam_config["image_size"])
    cam_config["intrinsics"] = list(cam_config["intrinsics"])
    cam_config["position"] = list(cam_config["position"])
    cam_config["rotation"] = list(cam_config["rotation"])
    cam_config["zrange"] = list(cam_config["zrange"])
    
    ee_trans = [0, 0, 0]
    ee_rot = R.from_matrix([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]).as_quat().tolist()
    
    partnr = Partnr.make(name="partnr", rate=rate_partnr, cam_config=[cam_config], ee_trans=ee_trans, ee_rot=ee_rot, debug=False)
    graph.add(partnr)
    graph.connect(source=speech.sensors.speech, target=partnr.inputs.speech)
    graph.connect(source=partnr.outputs.pick_pos, observation="pick_pos")
    graph.connect(source=partnr.outputs.pick_orn, observation="pick_orn")
    graph.connect(source=partnr.outputs.place_pos, observation="place_pos")
    graph.connect(source=partnr.outputs.place_orn, observation="place_orn")

    # Create camera
    from eagerx_demo.realsense.objects import RealSense
    import numpy as np

    image_size = RealSenseD415.CONFIG[0]["image_size"]
    focal_len = RealSenseD415.CONFIG[0]["intrinsics"][0]
    znear, zfar = RealSenseD415.CONFIG[0]["zrange"]
    fovh = (image_size[0] / 2) / focal_len
    fovh = 180 * np.arctan(fovh) * 2 / np.pi

    import eagerx_interbotix
    import os

    cam = RealSense.make(
        name="d435",
        rate=rate_cam,
        states=[],
        mode="rgbd",
        render_shape=list(image_size),
        base_pos=list(RealSenseD415.front_position),
        base_or=list(RealSenseD415.front_rotation),
        urdf=os.path.dirname(eagerx_interbotix.__file__) + "/camera/assets/realsense2_d435.urdf",
        optical_link="camera_bottom_screw_frame",
        calibration_link="camera_bottom_screw_frame",
        fov=float(fovh),
        near_val=float(znear),
        far_val=float(zfar),
    )
    graph.add(cam)

    # Connect
    graph.connect(source=cam.sensors.color, target=partnr.inputs.color)
    graph.connect(source=cam.sensors.depth, target=partnr.inputs.depth)
    graph.render(source=cam.sensors.color, rate=rate_cam, encoding="rgb")

    # graph.gui()

    # Create backend
    # from eagerx.backends.single_process import SingleProcess
    # backend = SingleProcess.make()
    from eagerx.backends.ros1 import Ros1  # todo: why does this not work?
    backend = Ros1.make()

    # Define engines
    from eagerx_pybullet.engine import PybulletEngine

    engine = PybulletEngine.make(rate=rate_engine, gui=False, egl=False, sync=True, real_time_factor=rtf)

    # from eagerx_reality.engine import RealEngine
    # engine = RealEngine.make(rate=rate_engine, sync=True)

    # Add Dummy object 'task' with a single EngineState that creates a task (if the engine is a PybulletEngine)
    if engine.config.entity_id == "eagerx_pybullet.engine/PybulletEngine":
        from eagerx_demo.task.enginestates import TaskState
        from eagerx.core.space import Space
        task_es_name = "task"
        task_es = TaskState.make(workspace=[0.2, 0.5, -0.3, 0.3])
        engine.add_object(task_es_name, urdf=None)
        task_es_space = Space(low=0, high=1, shape=(), dtype="int64")  # var that specifies the task.
        engine._add_engine_state(task_es_name, "reset", task_es, task_es_space.to_dict())

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
        reset_position=[0, -0.91435647,  0.85219240,  0, 1.6239657, 0],  # Position of the arm when reset (out-of-view)
        reset_gripper=[1.],  # Gripper position when reset (open)
    )

    # Evaluate
    action_space = env.action_space
    action = action_space.sample()
    for eps in range(5000):
        print(f"Episode {eps}")
        obs, info = env.reset()
        done = False
        while not done:
            obs, reward, terminated, truncated, info = env.step(obs)
            done = terminated or truncated