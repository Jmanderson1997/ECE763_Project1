from os import path 


def get_util_dir():
    return path.dirname(path.realpath(__file__))


def get_project_dir():
    return path.dirname(get_util_dir())


def get_data_dir():
    return path.join(get_project_dir(), 'data')


def get_fold_dir():
    return path.join(get_data_dir(), 'FDDB-folds')


def get_dataset_dir():
    return path.join(get_project_dir(), 'dataset')


def get_p1_pickle_folder():
    return path.join(get_project_dir(), 'project1_pickle')

def get_p2_pickle_folder():
    return path.join(get_project_dir(), 'project2_pickle')