import eagerx
import eagerx.core.register as register
from eagerx.core.specs import ObjectSpec
from eagerx.core.graph_engine import EngineGraph
from eagerx_pybullet.engine import PybulletEngine
from eagerx_reality.engine import RealEngine
from eagerx_demo.speech_recorder.node import SpeechInput


class SpeechRecorder(eagerx.Object):
    @classmethod
    @register.sensors(speech=eagerx.Space(dtype="uint8", shape=(280,), low=0, high=255))
    @register.engine_states()
    def make(
        cls,
        name: str,
        rate=10,
        base_pos=None,
        base_or=None,
        audio_device=None,
        type_commands=False,
        device="cpu",
        ckpt="base.en",
        prompt=None,
        test=False,
    ) -> ObjectSpec:
        """Make a spec to initialize a speech to speech object.

        :param name: Name of the object (topics are placed within this namespace).
        :param rate: The default rate at which all sensors run. Can be modified via the spec API.
        """
        spec = cls.get_specification()

        spec.config.name = name
        spec.config.rate = rate
        spec.config.sensors = ["speech"]
        spec.sensors.speech.rate = rate
        spec.config.base_pos = base_pos if isinstance(base_pos, list) else [0, 0, 1]
        spec.config.base_or = base_or if isinstance(base_or, list) else [0, 0, 0, 1]
        spec.config.audio_device = audio_device
        spec.config.type_commands = type_commands
        spec.config.device = device
        spec.config.ckpt = ckpt
        spec.config.prompt = prompt
        spec.config.test = test
        return spec

    @staticmethod
    @register.engine(PybulletEngine)
    def pybullet_engine(spec: ObjectSpec, graph: EngineGraph):
        import pybullet_data

        spec.engine.urdf = "" #%s/%s.urdf" % (pybullet_data.getDataPath(), "cube_small")
        spec.engine.basePosition = spec.config.base_pos
        spec.engine.baseOrientation = spec.config.base_or
        spec.engine.fixed_base = True
        spec.engine.self_collision = False

        recorder = SpeechInput.make(
            "speech_recorder",
            rate=spec.config.rate,
            audio_device=spec.config.audio_device,
            type_commands=spec.config.type_commands,
            device=spec.config.device,
            ckpt=spec.config.ckpt,
            prompt=spec.config.prompt,
            test=spec.config.test,
        )
        graph.add(recorder)
        graph.connect(source=recorder.outputs.speech, sensor="speech")

    @staticmethod
    @register.engine(RealEngine)
    def real_engine(spec: ObjectSpec, graph: EngineGraph):
        recorder = SpeechInput.make(
            "speech_recorder",
            rate=spec.config.rate,
            audio_device=spec.config.audio_device,
            type_commands=spec.config.type_commands,
            device=spec.config.device,
            ckpt=spec.config.ckpt,
            prompt=spec.config.prompt,
            test=spec.config.test,
        )
        graph.add(recorder)
        graph.connect(source=recorder.outputs.speech, sensor="speech")
