import os
from dotenv import load_dotenv
from project import get_absolute_path
import librosa
import numpy as np
import IPython.display as ipd


class AudioProcess:
    def __init__(self, audio_folder=None, processed_folder=None, sampling_rate=None):
        load_dotenv()

        self.audio_folder = get_absolute_path(audio_folder or os.getenv("AUDIO_FOLDER"))
        self.processed_folder = get_absolute_path(
            processed_folder or os.getenv("PROCESSED_FOLDER")
        )
        self.sampling_rate = sampling_rate or int(os.getenv("SAMPLING_RATE"))

        self.paths = self.get_audio_paths(self.audio_folder)
        self.melspectrograms = {"class_names": [], "data": []}

    def get_audio_paths(self, audio_folder):
        paths = {}
        for root, _, files in os.walk(audio_folder):
            if root != audio_folder:
                name = os.path.basename(root)
                for file in files:
                    if name in paths.keys():
                        paths[name].append(os.path.join(root, file))
                    else:
                        paths[name] = [os.path.join(root, file)]
        return paths

    def get_sample(self, audio_path) -> np.ndarray:
        return librosa.load(audio_path, sr=self.sampling_rate)[0]

    def get_audio(self, sample) -> ipd.Audio:
        return ipd.Audio(sample, rate=self.sampling_rate)

    def play_audio(self, input):
        sample = input
        if isinstance(input, str):
            sample = self.get_sample(input)
        ipd.display(self.get_audio(sample))