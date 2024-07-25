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
        self.melspectrograms = None

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

    def set_melspectrograms(self):
        melspectrograms = {
            "class_names": [],
            "data": [],
        }

        for class_name, paths in self.paths.items():
            melspectrograms["class_names"].append(class_name)
            for path in paths:
                sample = self.get_sample(path)
                melspectrogram = librosa.feature.melspectrogram(
                    y=sample, sr=self.sampling_rate
                )
                melspectrogram = librosa.power_to_db(melspectrogram, ref=np.max)
                melspectrograms["data"].append(
                    {
                        "label": melspectrograms["class_names"].index(class_name),
                        "spec": melspectrogram,
                    }
                )

        self.melspectrograms = melspectrograms
        return melspectrograms

    def get_melspectrograms(self):
        if not self.melspectrograms:
            self.set_melspectrograms()
        return self.melspectrograms

    def get_balanced_data(self):
        sample_count = {key: 0 for key in self.paths}
        min_sample = min({key: len(self.paths[key]) for key in self.paths}.values())

        balanced_data = {
            "class_names": self.melspectrograms["class_names"],
            "data": [],
        }

        for data in self.melspectrograms["data"]:
            actual_class = self.melspectrograms["class_names"][data["label"]]
            sample_count[actual_class] += 1
            if sample_count[actual_class] <= min_sample:
                balanced_data["data"].append(data)
        return balanced_data

    def get_first_10_melspectrograms(self):
        sample_count = {key: 0 for key in self.paths}

        test_data = {
            "class_names": self.melspectrograms["class_names"],
            "data": [],
        }

        for data in self.melspectrograms["data"]:
            actual_class = self.melspectrograms["class_names"][data["label"]]
            sample_count[actual_class] += 1
            if sample_count[actual_class] <= 10:
                test_data["data"].append(data)
        return test_data

    def import_melspectrograms(self, spectrograms):
        self.melspectrograms = spectrograms
