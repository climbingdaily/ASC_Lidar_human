import os


def get_one_path_by_suffix(dirname, suffix):
    filenames = list(filter(lambda x: x.endswith(suffix), os.listdir(dirname)))
    assert len(filenames) > 0
    return os.path.join(dirname, filenames[0])


def get_index(filename):
    basename = os.path.basename(filename)
    return int(os.path.splitext(basename)[0])


def get_sorted_filenames_by_index(dirname, isabs=True):
    filenames = os.listdir(dirname)
    filenames = sorted(os.listdir(dirname), key=lambda x: get_index(x))
    if isabs:
        filenames = [os.path.join(dirname, filename) for filename in filenames]
    return filenames


def remove_unnecessary_frames(dirname, indexes):
    filenames = os.listdir(dirname)
    assert len(filenames) > 0
    for filename in filenames:
        if get_index(filename) not in indexes:
            os.remove(os.path.join(dirname, filename))
