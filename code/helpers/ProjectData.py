import os
from project import get_absolute_path
from dotenv import load_dotenv


class ProjectData:
    def __init__(self, audioFolder=None, processedFolder=None):

        load_dotenv()

        self.audioFolder = get_absolute_path(audioFolder or os.getenv("AUDIO_FOLDER"))
        self.processedFolder = get_absolute_path(
            processedFolder or os.getenv("PROCESSED_FOLDER")
        )
        self.audioPaths = None
        self.dataPaths = None

        self.set_audio_paths()

    def set_audio_paths(self):
        paths = {}
        for root, _, files in os.walk(self.audioFolder):
            if root != self.audioFolder:
                name = os.path.basename(root)
                for file in files:
                    if name in paths.keys():
                        paths[name].append(os.path.join(root, file))
                    else:
                        paths[name] = [os.path.join(root, file)]
        self.audioPaths = paths

    def set_data_paths(self):
        dataPaths = {}
        audioLabel = None
        pathKeys = list(self.audioPaths.keys())
        for root, _, files in os.walk(self.processedFolder):
            if root != self.processedFolder:
                if len(files) == 0:
                    audioLabel = pathKeys.index(os.path.basename(root))
                else:
                    for file in files:
                        dataType = file.split(".")[0]
                        if dataType in dataPaths.keys():
                            dataPaths[dataType].append(
                                {audioLabel: os.path.join(root, file)}
                            )
                        else:
                            dataPaths[dataType] = [
                                {audioLabel: os.path.join(root, file)}
                            ]
        self.dataPaths = dataPaths

    def get_audio_paths(self, class_name=None, indexes=None):
        if self.audioPaths == None:
            self.set_audio_paths()
        if indexes == None:
            return {
                key: self.audioPaths[key]
                for key in self.audioPaths.keys()
                if class_name == None or key in class_name
            }
        else:
            return {
                key: [self.audioPaths[key][i] for i in indexes]
                for key in self.audioPaths.keys()
                if class_name == None or key in class_name
            }

    def get_data_paths(self, data_types=None, indexes=None):
        if self.dataPaths == None:
            self.set_data_paths()
        if indexes == None:
            return {
                key: self.dataPaths[key]
                for key in self.dataPaths.keys()
                if data_types == None or key in data_types
            }
        else:
            return {
                key: [self.dataPaths[key][i] for i in indexes]
                for key in self.dataPaths.keys()
                if data_types == None or key in data_types
            }

    def get_data_types(self):
        return list(self.dataPaths.keys())
