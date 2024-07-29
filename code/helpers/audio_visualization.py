import json
import numpy as np
import matplotlib.pyplot as plt
import librosa.display


def load_mfcc_from_json(json_file_path):
    """Load MFCC data from a JSON file.

    Args:
        json_file_path (str): Path to the JSON file containing MFCC data.

    Returns:
        np.ndarray: MFCC data as a numpy array.
    """
    with open(json_file_path, "r") as file:
        data = json.load(file)
        mfcc = np.array(data["mfcc"])
    return mfcc


def show_mfcc(mfcc, sampling_rate, title="MFCC"):
    """Show MFCC plot.

    Args:
        mfcc (np.ndarray): MFCC data.
        sampling_rate (int): Sampling rate for the plot.
    """
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, sr=sampling_rate, x_axis="time")
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def show_mfcc_from_json(json_file_path, sampling_rate, title="MFCC"):
    """Show MFCC plot from a JSON file.

    Args:
        json_file_path (str): Path to the JSON file containing MFCC data.
        sampling_rate (int): Sampling rate for the plot.
    """
    mfcc = load_mfcc_from_json(json_file_path)
    show_mfcc(mfcc, sampling_rate, title)


def load_melspectrogram_from_json(json_file_path):
    """Load Mel Spectrogram data from a JSON file.

    Args:
        json_file_path (str): Path to the JSON file containing Mel Spectrogram data.

    Returns:
        np.ndarray: Mel Spectrogram data as a numpy array.
    """
    with open(json_file_path, "r") as file:
        data = json.load(file)
        melSpectogram = np.array(data["melSpectrogram"])
    return melSpectogram


def show_melspectrogram_from_json(
    json_file_path, sampling_rate, title="Mel Spectrogram"
):
    """Show Mel Spectrogram plot from a JSON file.

    Args:
        json_file_path (str): Path to the JSON file containing Mel Spectrogram data.
        sampling_rate (int): Sampling rate for the plot.
    """
    melspectrogram = load_melspectrogram_from_json(json_file_path)
    show_melspectrogram(melspectrogram, sampling_rate, title)


def show_melspectrogram(melspectrogram, sampling_rate, title="Mel Spectrogram"):
    """Show Mel Spectrogram plot.

    Args:
        melspectrogram (np.ndarray): Mel Spectrogram data.
        sampling_rate (int): Sampling rate for the plot.
    """
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        melspectrogram, sr=sampling_rate, x_axis="time", y_axis="mel"
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.tight_layout()
    plt.show()
