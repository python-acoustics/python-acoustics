import os


def data_path(file_name):
    parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return parent + '/data/' + file_name
