import cv2
import yaml
import os.path as osp

def load_yaml(config_file):
    config_obj = {}
    if osp.exists(config_file):
        with open(config_file, 'r') as f:
                config_obj = yaml.safe_load(f)
    return config_obj