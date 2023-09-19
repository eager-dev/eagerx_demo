import eagerx
from eagerx.core.specs import NodeSpec
from eagerx_demo.utils import string_to_uint8

from threading import Thread
from typing import Any
from pathlib import Path
import queue
import sys
import numpy as np
import tempfile
import whisper
import torch


ROOT = Path(__file__).parent.parent.parent


class SpeechInput(eagerx.EngineNode):
    @classmethod
    def make(
        cls,
        name: str,
        rate: float,
        process: int = eagerx.process.NEW_PROCESS,
        color: str = "cyan",
        audio_device: str = None,
        audio_dir: str = None,
        sample_rate: int = None,
        channels: int = 1,
        subtype: str = None,
        type_commands: bool = False,
        device: str = "cpu",
        ckpt: str = "base.en",
        prompt: str = None,
        test: bool = False,
    ):
        spec = cls.get_specification()
        spec.config.update(
            name=name,
            rate=rate,
            process=process,
            color=color,
            audio_device=audio_device,
            audio_dir=audio_dir,
            sample_rate=sample_rate,
            channels=channels,
            subtype=subtype,
            type_commands=type_commands,
            device=device,
            ckpt=ckpt,
            prompt=prompt,
            test=test,
            inputs=["tick"],
            outputs=["speech"],
        )
        return spec

    def initialize(self, spec: NodeSpec, simulator: Any):
        self.input = None
        self.correct = False
        self.recording = False
        self.typing = False
        self.prompt = spec.config.prompt
        audio_dir = spec.config.audio_dir or str(ROOT / "data" / "audio")
        Path(audio_dir).mkdir(parents=True, exist_ok=True)
        self.audio_path = tempfile.mktemp(prefix="rec_", suffix=".wav", dir=audio_dir)
        self.audio_device = spec.config.audio_device

        # In type_commands mode we don't record audio
        self.type_commands = spec.config.type_commands
        self.test = spec.config.test
        if self.type_commands or self.test:
            self.device_info = {}
            self.sample_rate = 16000
        else:
            import sounddevice as sd

            self.device_info = sd.query_devices(self.audio_device, "input")
            self.sample_rate = spec.config.sample_rate or int(self.device_info["default_samplerate"])

        self.device = spec.config.device
        self.model = whisper.load_model(spec.config.ckpt, device=torch.device(self.device))
        self.channels = spec.config.channels
        self.subtype = spec.config.subtype
        self.q = queue.Queue()
        if not self.test:
            thread = Thread(target=self._speech_recorder, daemon=True)
            thread.start()

    @eagerx.register.states()
    def reset(self):
        self.input = None
        self.correct = False
        self.speech = ""

        self.q = queue.Queue()
        print("#" * 80)
        print("hold r to record command")
        print("#" * 80)

    @eagerx.register.inputs(tick=eagerx.Space(dtype="int64"))
    @eagerx.register.outputs(speech=eagerx.Space(dtype="uint8", shape=(280,)))
    def callback(self, t_n: float, tick: Any = None):
        speech = ""
        if self.input is not None and self.correct:
            speech = self.input
            self.input = None
            self.correct = False

        speech = string_to_uint8(speech)
        return dict(speech=speech)

    def _speech_recorder(self):
        # Collect events until released
        from pynput.keyboard import Listener

        with Listener(on_press=self._on_press, on_release=self._on_release) as listener:
            listener.join()

    def _on_press(self, key):
        if not self.typing and not self.recording and self.input is None:
            if hasattr(key, "char") and key.char == ("r"):
                self.recording = True
                q = queue.Queue()
                # First delete the file if it already exists
                try:
                    Path(self.audio_path).unlink()
                except FileNotFoundError:
                    pass
                print("RECORD...")
                thread = Thread(target=self._soundfile_writer, daemon=True)
                thread.start()
        if not self.typing and not self.recording and self.input is not None and not self.correct:
            if hasattr(key, "char") and key.char == ("y"):
                self.correct = True
                print(f"Command {self.input} will be executed.")
                print("#" * 80)
                print("hold r to record")
                print("#" * 80)
            if hasattr(key, "char") and key.char == ("n"):
                self.input = None
                self.correct = False
                print("Sorry I misunderstood, please try again!")
                print("#" * 80)
                print("hold r to record")
                print("#" * 80)

    def _on_release(self, key):
        if not self.typing and self.recording:
            self.recording = False
            if hasattr(key, "char") and key.char == ("r"):
                print("Done RECORD")
                if self.type_commands:
                    self.typing = True
                    text = input("Enter command: ")
                    while text.startswith("r"):
                        text = text[1:]
                    self.typing = False
                    self.correct = True
                    result = dict(text=text)
                else:
                    kwargs = dict(initial_prompt=self.prompt)
                    if self.device == "cpu":
                        kwargs["fp16"] = False
                    result = self.model.transcribe(
                        self.audio_path,
                        **kwargs,
                    )
                self.input = result["text"]
                if "stop" in result["text"].lower():
                    self.correct = True
                else:
                    print(f"Is the following command correct?\n{result['text']}\n(y/n)")

    def _sd_callback(self, indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        self.q.put(indata.copy())

    def _soundfile_writer(self):
        import sounddevice as sd
        import soundfile as sf
        with sf.SoundFile(
            self.audio_path, mode="x", samplerate=self.sample_rate, channels=self.channels, subtype=self.subtype
        ) as file:
            if self.type_commands:
                while self.recording:
                    file.write(np.asarray(1, dtype="int32"))
            else:
                with sd.InputStream(
                    samplerate=self.sample_rate, device=self.audio_device, channels=self.channels, callback=self._sd_callback
                ):
                    while self.recording:
                        file.write(self.q.get())
