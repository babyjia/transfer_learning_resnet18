import torch
import numpy as np
from PIL import Image
import os

def mkdirs(paths):
    #if many paths then mkdir one by one
    if isinstance(paths,list) and not isinstance(paths,str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)