#####################################################################
# efficientnet.py
#
# Dev. Dongwon Paek
# Description: Main source code of SqueezeNet
#####################################################################

import os, shutil, time
import numpy as np

import torch
from torch.backends import cudnn
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from model import EfficientNet

from hyperparameter import *

def load_dataset_train():
    data_path = '/home/bearpaek/data/datasets/lplSmall/train/'
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=torchvision.transforms.ToTensor()
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True
    )
    return train_loader

def load_dataset_test():
    data_path = '/home/bearpaek/data/datasets/lplSmall/validation/'
    test_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=torchvision.transforms.ToTensor()
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True
    )
    return test_loader

def save_model(model, e):
    model_save_path = './models'
    path = os.path.join(
        model_save_path,
        '{}.pth'.format(e)
    )
    torch.save(model.state_dict(), path)
    
def train(epoch):
    global avg_loss
    correct = 0
    model.train()
    for b_idx, (data, targets) in enumerate(load_dataset_train()):
        data, targets = data.cuda(), targets.cuda()
        # convert the data and targets into Variable and cuda form
        data, targets = Variable(data), Variable(targets)

        # train the network
        optimizer.zero_grad()
        scores = model.forward(data)
        scores = scores.view(config.batch_size, config.num_classes)
        loss = F.nll_loss(scores, targets)

        # compute the accuracy
        pred = scores.data.max(1)[1]
        correct += pred.eq(targets.data).cuda().sum()

        avg_loss.append(loss.data)
        loss.backward()
        optimizer.step()

        if b_idx % config.loss_log_step == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (b_idx+1) * len(data), len(load_dataset_train().dataset),
                100. * (b_idx+1)*len(data) / len(load_dataset_train().dataset), loss.data))
        
    # now that the epoch is completed plot the accuracy
    train_accuracy = correct / float(len(load_dataset_train().dataset))
    print("training accuracy ({:.2f}%)".format(100*train_accuracy))
    return (train_accuracy*100.0)


def val():
    global best_accuracy
    correct = 0
    model.eval()
    for idx, (data, target) in enumerate(load_dataset_test()):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        # do the forward pass
        score = model.forward(data)
        pred = score.data.max(1)[1] # got the indices of the maximum, match them
        correct += pred.eq(target.data).cpu().sum()

    print("predicted {} out of {}".format(correct, 73*64))
    val_accuracy = correct / (73.0*64.0) * 100
    print("accuracy = {:.2f}".format(val_accuracy))

    # now save the model if it has better accuracy than the best model seen so forward
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        # save the model
        torch.save(model.state_dict(),'efficientnet_torch.pth')
    return val_accuracy

def test():
    # load the best saved model
    weights = torch.load('efficientnet_torch.pth')
    model.load_state_dict(weights)
    model.eval()

    test_correct = 0
    total_examples = 0
    accuracy = 0.0
    for idx, (data, target) in enumerate(load_dataset_test()):
        if idx < 73:
            continue
        total_examples += len(target)
        data, target = Variable(data), Variable(target)
        data, target = data.cuda(), target.cuda()

        scores = model(data)
        pred = scores.data.max(1)[1]
        test_correct += pred.eq(target.data).cpu().sum()
    print("Predicted {} out of {} correctly".format(test_correct, total_examples))
    return 100.0 * test_correct / (float(total_examples))
    

if __name__ == '__main__':
    torch.cuda.device(0)
    model = EfficientNet(1.0, 1.0)
    
    config = CONFIG()
    
    model = model.cuda()

    avg_loss = list()
    best_accuracy = 0.0

    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=config.momentum, weight_decay=config.weight_decay)
    
    train_acc, val_acc = list(), list()

    for i in range(1, EPOCHS+1):
        train_acc.append(train(i))
        val_acc.append(val())
        save_model(model, i)
        print(train_acc)
        print(val_acc)