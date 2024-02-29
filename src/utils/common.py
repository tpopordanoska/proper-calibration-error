import os
import pickle

import torch


def create_folder(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        print(f"Directory {path} already exists.")
    except OSError:
        print(f"Creation of the directory {path} failed.")
    else:
        print(f"Successfully created the directory {path}.")

    return path


def load_pickle(path, **kwargs):
    with open(path, 'rb') as fp:
        return pickle.load(fp, **kwargs)


def dump_pickle(path, data, **kwargs):
    with open(path, 'wb') as fp:
        pickle.dump(data, fp, **kwargs)


def str_to_num(integers):
    strings = [str(integer) for integer in integers]
    a_string = "".join(strings)
    return a_string


def get_scores_and_labels(preds_dict, logits_key, labels_key):
    EPS = 1e-7

    logits = preds_dict[logits_key]
    labels = preds_dict[labels_key]

    scores = torch.softmax(logits, dim=1)
    scores = torch.clamp(scores, min=EPS, max=1 - EPS)

    return scores, labels
