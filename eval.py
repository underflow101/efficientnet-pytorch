##########################################################################
# eval.py
# Dev. Dongwon Paek
#
# Description: Validation test of PyTorch model file.
##########################################################################

import os, sys, time
import cv2
import numpy as np
from threading import Thread
import importlib.util
from PIL import Image

from cv2 import getTickCount, getTickFrequency

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from model import EfficientNet

from hyperparameter import *

def main():
    device = torch.device('cpu')
    model = EfficientNet(1.0, 1.0)
    weights = torch.load('efficientnet_torch.pth', map_location=torch.device('cpu'))
    model.load_state_dict(weights)
    model.eval()
    
    config = CONFIG()
    
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=config.momentum, weight_decay=config.weight_decay)
    
    frame_rate_calc = 1
    freq = getTickFrequency()
    
    image_path = []
    
    writing = 0
    phoneWithHand = 0
    others = 0
    sleep = 0

    image_path.append("./others/")
    image_path.append("./phoneWithHand/")
    image_path.append("./sleep/")
    image_path.append("./writing/")
    
    for j in range(0, 4):
        sumsum = 0
        others = 0
        phoneWithHand = 0
        sleep = 0
        writing = 0
        for i in range(0, 100):
            image = Image.open(image_path[j] + str(i) + '.jpg')
            trans = torchvision.transforms.ToTensor()
            image = trans(image)
            data = image.unsqueeze(0)
            t1 = getTickCount()
            scores = model(data)
            t2 = getTickCount()
            time1 = (t2 - t1) / freq
            frame_rate_calc = 1 / time1
            sumsum += frame_rate_calc
            pred = scores.data.max(1)[1]

        sumsum /= 100
        print('others: ', end='')
        print(others)
        print('phoneWithHand: ', end='')
        print(phoneWithHand)
        print('writing: ', end='')
        print(writing)
        print('sleep: ', end='')
        print(sleep)
        print('Average FPS: ', end='')
        print(sumsum)
    
if __name__ == '__main__':
    main()
