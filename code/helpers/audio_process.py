import os
from dotenv import load_dotenv
import librosa
import numpy as np
import IPython.display as ipd
from json import dump
from project import get_absolute_path
from scipy.signal import butter, lfilter


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

        self.data = {}
        self.sample = None

    def set_sample(self, sample=None):
        if sample:
            self.sample = sample
        else:
            self.sample = self.get_sample(self.audio_path)

    def get_visualization(self, types=None, sample=None):

        if types == None:
            types = ["melSpectrogram", "mfcc"]

        visualization = {}
        for type in types:
            if type == "melSpectrogram":
                visualization["melSpectrogram"] = self.get_melspectrogram(sample=sample)
            elif type == "mfcc":
                visualization["mfcc"] = self.get_mfcc(sample=sample)
            else:
                print(f"{type} - This type is not defined")
        return visualization

    def get_data(self, augmentations=None, types=None):
        if not self.data:
            self.create_data()

        if augmentations == None:
            augmentations = self.data.keys()

        if types == None:
            result = {
                aug: self.data[aug] for aug in augmentations if aug in self.data.keys()
            }
        else:
            result = {
                aug: {
                    key: self.data[key] for key in types if key in self.data[aug].keys()
                }
                for aug in augmentations
                if aug in self.data.keys()
            }
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
        if sample is None:
            sample = self.sample
        if not sampling_rate:
            sampling_rate = self.sampling_rate

        melspectrogram = librosa.feature.melspectrogram(y=sample, sr=sampling_rate)
        melspectrogram = librosa.power_to_db(melspectrogram, ref=np.max)
        return melspectrogram

    def get_mfcc(self, sample=None, sampling_rate=None):
        if sample is None:
            sample = self.sample
        if not sampling_rate:
            sampling_rate = self.sampling_rate

        mfcc = librosa.feature.mfcc(y=sample, sr=sampling_rate, n_mfcc=13)
        return mfcc

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

    def add_orginal_data(self, types=None):
        if self.sample is None:
            self.sample = self.get_sample(self.audio_path)
        self.data["original"] = self.get_visualization(types=types)

    def add_noise_data(self, noise_factor=0.005, types=None):
        if self.sample is None:
            self.sample = self.get_sample(self.audio_path)

        noise = np.random.randn(len(self.sample))
        augmented_sample = self.sample + noise_factor * noise
        self.data["noise"] = self.get_visualization(
            sample=augmented_sample, types=types
        )

    def add_echo(self, delay=0.2, decay=0.5, types=None):
        if self.sample is None:
            self.sample = self.get_sample(self.audio_path)

        echo_sample = np.zeros_like(self.sample)
        delay_samples = int(delay * self.sampling_rate)
        for i in range(delay_samples, len(self.sample)):
            echo_sample[i] = self.sample[i] + decay * self.sample[i - delay_samples]
        self.data["echo"] = self.get_visualization(sample=echo_sample, types=types)

    def add_frequency_filter_data(self, lowcut=500.0, highcut=15000.0, types=None):
        if self.sample is None:
            self.sample = self.get_sample(self.audio_path)

        nyquist = 0.5 * self.sampling_rate
        low = lowcut / nyquist
        high = highcut / nyquist

        if not (0 < low < 1):
            lowcut = nyquist * 0.1
            low = lowcut / nyquist
        if not (0 < high < 1):
            highcut = nyquist * 0.9
            high = highcut / nyquist

        b, a = butter(1, [low, high], btype="band")
        filtered_sample = lfilter(b, a, self.sample)
        self.data["frequency_filter"] = self.get_visualization(
            sample=filtered_sample, types=types
        )

    def add_delay_data(self, delay=0.2, types=None):
        if self.sample is None:
            self.sample = self.get_sample(self.audio_path)

        delay_samples = int(delay * self.sampling_rate)
        delayed_sample = np.zeros_like(self.sample)
        for i in range(delay_samples, len(self.sample)):
            delayed_sample[i] = self.sample[i - delay_samples]
        self.data["delay"] = self.get_visualization(sample=delayed_sample, types=types)

    def add_reverb_data(self, reverb_factor=0.5, types=None):
        if self.sample is None:
            self.sample = self.get_sample(self.audio_path)

        reverb_sample = np.convolve(
            self.sample,
            np.ones(int(self.sampling_rate * reverb_factor)) / self.sampling_rate,
            mode="full",
        )
        self.data["reverb"] = self.get_visualization(sample=reverb_sample, types=types)
