from os import path 


def get_proj_dir():
    return path.dirname(path.realpath(__file__)) 


def get_fold_dir():
    return path.join(get_proj_dir(), 'FDDB-folds')


def get_pickle_folder():
    return path.join(get_proj_dir(), 'pickle')