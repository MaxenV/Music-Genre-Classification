from os import path


def get_project_root_path():
    return path.abspath("../../")


def get_absolute_path(relative_path):
    project_root_path = get_project_root_path()
    return path.join(project_root_path, relative_path)
