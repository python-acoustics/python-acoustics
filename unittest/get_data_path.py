import os


def data_path():
    parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(parent, 'data/')
