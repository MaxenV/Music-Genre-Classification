import os
from dotenv import load_dotenv
import librosa
import numpy as np
import IPython.display as ipd
from json import dump
from project import get_absolute_path


class AudioProcess:
    def __init__(
        self,
        audio_path,
        class_name,
        sampling_rate=None,
        sample_length=None,
        processed_folder=None,
    ):
        """Initialize the AudioProcess class

        Args:
            audio_path (str): Path of the audio file.
            class_name (str): Class name of the audio file.
            sampling_rate (int, optional): Sampling rate for audio processing. Defaults to None.
            sample_length (int, optional): Length of the audio sample. Defaults to None.
        """
        load_dotenv()

        self.sampling_rate = sampling_rate or int(os.getenv("SAMPLING_RATE"))
        self.sample_length = sample_length or int(os.getenv("SAMPLE_LENGTH"))
        self.processed_folder = processed_folder or get_absolute_path(
            os.getenv("PROCESSED_FOLDER")
        )

        self.audio_path = audio_path
        self.audio_name = os.path.basename(audio_path)
        self.class_name = class_name

        self.data = None
        self.sample = None

    def create_data(self, types=["melSpectrogram"]):
        self.data = {}
        self.sample = self.get_sample(self.audio_path)

        for type in types:
            if type == "melSpectrogram":
                self.data["melSpectrogram"] = self.get_melspectrogram()
            else:
                print(f"{type} - This type is ont defined")

    def get_data(self, types=None):
        if not self.data:
            self.create_data()

        if types == None:
            return self.data.copy()

        result = {key: self.data[key] for key in types if key in self.data.keys()}
        return result

    def standardize_sample_length(self, sample, sample_length):
        if len(sample) > sample_length:
            sample = sample[:sample_length]
        else:
            sample = np.pad(
                sample,
                (0, sample_length - len(sample)),
                "constant",
                constant_values=(0),
            )
        return sample

    def get_sample(self, audio_path) -> np.ndarray:
        sample, _ = librosa.load(audio_path, sr=self.sampling_rate)
        sample = self.standardize_sample_length(sample, self.sample_length)
        return sample

    def get_audio(self, sample, sampling_rate=None) -> ipd.Audio:
        if not sampling_rate:
            sampling_rate = self.sampling_rate
        return ipd.Audio(sample, rate=sampling_rate)

    def play_audio(self, input):
        sample = input
        if isinstance(input, str):
            sample = self.get_sample(input)
        ipd.display(self.get_audio(sample))

    def get_melspectrogram(self, sample=None, sampling_rate=None):
        if not sample:
            sample = self.sample
        if not sampling_rate:
            sampling_rate = self.sampling_rate

        melspectrogram = librosa.feature.melspectrogram(y=sample, sr=sampling_rate)
        melspectrogram = librosa.power_to_db(melspectrogram, ref=np.max)
        return melspectrogram

    def save_data(self, types=None, processed_folder=None):
        if not self.data:
            print("No data to save")

        if not types:
            types = self.data.keys()
        if not processed_folder:
            processed_folder = self.processed_folder

        joined_types = str.join("_", types)

        class_folder_path = os.path.join(processed_folder, self.class_name)
        audio_folder_path = os.path.join(
            class_folder_path, self.audio_name.replace(".", "_")
        )
        file_path = os.path.join(
            audio_folder_path,
            f"{joined_types}.json",
        )

        os.makedirs(class_folder_path, exist_ok=True)
        os.makedirs(audio_folder_path, exist_ok=True)

        with open(file_path, "w") as file:
            to_json = {key: self.data[key].tolist() for key in types}
            dump(to_json, file)
