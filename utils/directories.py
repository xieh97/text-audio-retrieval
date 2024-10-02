import os


def get_dataset_dir():
    data_dir = os.getenv('DATA_HOME', '~/Audio_Datasets')
    return make_if_not_exits(data_dir)


def get_persistent_cache_dir():
    data_dir = os.getenv('DATA_HOME', '~/Audio_Datasets')
    return make_if_not_exits(data_dir)


def make_if_not_exits(path):
    os.makedirs(path, exist_ok=True)
    return path
