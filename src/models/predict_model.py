import sys
import argparse

import torch
from torch import nn, optim
import torch.nn.functional as F
import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

os.chdir("C:/Users/annas/OneDrive/Dokumente/DTU/8_semester/3weeks_MLops/MLops_exercises_organized/")

MODEL_PATH = "src\\models\\model.pth"
DATA_PATH = "data\\processed\\"
FIGURE_PATH = "reports\\figures\\"


from torchvision import datasets, transforms

from model import MyAwesomeModel
import numpy as np

import matplotlib.pyplot as plt
#import plotext.plot as plx


class Predict():
    def __init__(self):
        self.predict()
    
    def predict(self, input):
        print("Predict class for a given Image")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--load_model_from', default=MODEL_PATH)
        # add any additional argument that you want
        
        
        model = MyAwesomeModel(10)
        dict_ = torch.load(MODEL_PATH)
        model.load_state_dict(dict_)
        
        
        # turn off gradients for the purpose of speeding up the code
        predicted_classes = []
        with torch.no_grad():
            for images in input: # with batch size 64
                ps = torch.exp(model(images))
                top_p, top_class = ps.topk(1, dim=1)
                predicted_classes.append(top_class)
            
        return predicted_classes