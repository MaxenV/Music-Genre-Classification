import os
from dotenv import load_dotenv
from project import get_absolute_path


class AudioProcess:
    def __init__(self, audio_folder=None, processed_folder=None, sampling_rate=None):
        load_dotenv()

        self.audio_folder = get_absolute_path(audio_folder or os.getenv("AUDIO_FOLDER"))
        self.processed_folder = get_absolute_path(
            processed_folder or os.getenv("PROCESSED_FOLDER")
        )
        self.sampling_rate = sampling_rate or int(os.getenv("SAMPLING_RATE"))

        self.paths = {}
        self.melspectrograms = {"class_names": [], "data": []}
