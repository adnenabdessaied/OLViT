import json
import os

def read_default_config():
    dirpath = os.path.dirname(__file__)
    path = os.path.join(dirpath, "default.json")
    with open(path) as config_file:
        config = json.load(config_file)
    return config

def read_config(path):
    with open(path) as config_file:
        config = json.load(config_file)
    return config

def update_nested_dicts(old_dict, update_dict):
    for key in update_dict:
        if key in old_dict:
            old_dict[key].update(update_dict[key])
        else:
            old_dict[key] = update_dict[key]
    return old_dict



 