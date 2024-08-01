import numpy as np
from keras.utils import Sequence
import numpy as np
from keras.utils import to_categorical
from keras.utils import to_categorical


class DataGenerator(Sequence):
    def __init__(
        self,
        file_paths,
        labels,
        mel_shape,
        mfcc_shape,
        num_classes,
        batch_size=32,
        shuffle=True,
    ):
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mel_shape = mel_shape
        self.mfcc_shape = mfcc_shape
        self.num_classes = num_classes
        self.on_epoch_end()

    def __len__(self):
        return len(self.file_paths) // self.batch_size

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        batch_file_paths = [self.file_paths[i] for i in indices]
        batch_labels = [self.labels[i] for i in indices]
        X, y = self.__data_generation(batch_file_paths, batch_labels)
        return X, y

    def on_epoch_end(self):
        self.indices = np.arange(len(self.file_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __data_generation(self, batch_file_paths, batch_labels):
        X_mel = np.empty((self.batch_size,) + self.mel_shape)
        X_mfcc = np.empty((self.batch_size,) + self.mfcc_shape)
        y = np.empty((self.batch_size, self.num_classes), dtype=int)

        for i, (file_path, label) in enumerate(zip(batch_file_paths, batch_labels)):
            data = load_json_data(file_path)
            X_mel[i,] = data["melSpectrogram"].reshape(self.mel_shape)
            X_mfcc[i,] = data["mfcc"].reshape(self.mfcc_shape)
            y[i] = to_categorical(label, num_classes=self.num_classes)

        return {"input_mel": X_mel, "input_mfccs": X_mfcc}, y

    def get_single_input(self):
        random_index = np.random.randint(len(self.file_paths))
        batch_index = random_index // self.batch_size
        index_within_batch = random_index % self.batch_size
        X, y = self.__getitem__(batch_index)
        return (
            {
                key: value[index_within_batch : index_within_batch + 1]
                for key, value in X.items()
            },
            y[index_within_batch : index_within_batch + 1],
        )
