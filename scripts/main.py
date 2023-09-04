import eagerx
import eagerx_interbotix
from eagerx.core.specs import EngineSpec
from datetime import datetime
import os


NAME = "OpenDR_demo"
LOG_DIR = os.path.dirname(eagerx_interbotix.__file__) + f"/../logs/{NAME}_{datetime.today().strftime('%Y-%m-%d-%H%M')}"


if __name__ == "__main__":
    eagerx.set_log_level(eagerx.WARN)

    MUST_LOG = False
    robot_type = "vx300s"
    rtf = 0
    rate_env = 10
    rate_speech = 10
    rate_xseries = 10
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
    # from eagerx_demo.speech_recorder.objects import SpeechRecorder
    #
    # speech = SpeechRecorder.make("speech", rate=rate_speech, audio_device=None, debug=False, device="cpu", ckpt="base.en", prompt=None)
    # graph.add(speech)
    # graph.connect(source=speech.sensors.speech, observation="speech")

    # Create camera
    from eagerx_interbotix.camera.objects import Camera

    cam = Camera.make(
        "cam",
        rate=rate_cam,
        sensors=["image"],
        urdf=os.path.dirname(eagerx_interbotix.__file__) + "/camera/assets/realsense2_d435.urdf",
        optical_link="camera_color_optical_frame",
        calibration_link="camera_bottom_screw_frame",
        camera_index=0,
    )
    graph.add(cam)

    # Connect
    graph.render(source=cam.sensors.image, rate=rate_cam, encoding="bgr")

    # Create backend
    from eagerx.backends.single_process import SingleProcess
    backend = SingleProcess.make()
    # from eagerx.backends.ros1 import Ros1  # todo: why does this not work?
    # backend = Ros1.make()

    # Define engines
    from eagerx_pybullet.engine import PybulletEngine
    engine = PybulletEngine.make(rate=rate_engine, gui=True, egl=True, sync=True, real_time_factor=rtf)

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
