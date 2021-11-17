import logging
import argparse


import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
# from ConvNet import ConvNet 
import argparse
import numpy as np


from loader import ModelLoader

def getArguments():
    """
        Argument parser. Read help for details
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelConfig", dest="modelConfig", help="Model Config JSON to be loaded")
    parser.add_argument("--testFraction", dest="testFraction", default=0.8, type=int, help="Fraction to be used for test dataset from total dataset")
    allArgs = parser.parse_args()
    return allArgs

def setupLogger():
    """
        Logger setup
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", '%m-%d-%Y %H:%M:%S')
    stdout = logging.StreamHandler(sys.stdout)
    stdout.setLevel(logging.INFO)
    stdout.setFormatter(formatter)
    logger.addHandler(stdout)
    logging.debug("Setting up logger completed")

def train(model, device, train_loader, optimizer, criterion, epoch, batch_size):
    '''
    Trains the model for an epoch and optimizes it.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    train_loader: dataloader for training samples.
    optimizer: optimizer to use for model parameter updates.
    criterion: used to compute loss for prediction and target 
    epoch: Current epoch to train for.
    batch_size: Batch size to be used.
    '''
    
    # Set model to train mode before each epoch
    model.train()
    
    # Empty list to store losses 
    losses = []
    correct = 0
    
    # Iterate over entire training samples (1 epoch)
    for batch_idx, batch_sample in enumerate(train_loader):
        import time
        # print("Batch starts: {}".format(time.time()))
        data, target = batch_sample
        
        # Push data/label to correct device
        data, target = data.to(device), target.to(device)
        
        # Reset optimizer gradients. Avoids grad accumulation (accumulation used in RNN).
        optimizer.zero_grad()
        
        # Do forward pass for current set of data
        import pdb
        pdb.set_trace()
        output = model(data)
        
        # ======================================================================
        # Compute loss based on criterion
        # ----------------- YOUR CODE HERE ----------------------
        #
        # Remove NotImplementedError and assign correct loss function.
        loss = criterion(output, target)
        
        # Computes gradient based on final loss
        loss.backward()
        
        # Store loss
        losses.append(loss.item())
        
        # Optimize model parameters based on learning rate and gradient 
        optimizer.step()
        
        # Get predicted index by selecting maximum log-probability
        pred = output.argmax(dim=1, keepdim=True)
        
        # ======================================================================
        # Count correct predictions overall 
        # ----------------- YOUR CODE HERE ----------------------
        #
        # Remove NotImplementedError and assign counting function for correct predictions.
        correct += countCorrectPredictions(pred, target)
        
    train_loss = float(np.mean(losses))
    train_acc = (correct / ((batch_idx+1) * batch_size)) * 100.0
    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        float(np.mean(losses)), correct, (batch_idx+1) * batch_size,
        100. * correct / ((batch_idx+1) * batch_size)))
    return train_loss, train_acc


if __name__ == "__main__":
    # Logger
    # setupLogger()
    
    # Argument Parser
    # allArgs = getArguments()
    # bla = ModelLoader("model_1.json")

    # ========================================================
    use_cuda = torch.cuda.is_available()
    
    # Set proper device based on cuda availability 
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Torch device selected: ", device)
    
    # Initialize the model and send to device 
    model = ModelLoader("model_1.json").to(device)
    
    # ======================================================================
    # Define loss function.
    # ----------------- YOUR CODE HERE ----------------------
    #
    # Remove NotImplementedError and assign correct loss function.
    # criterion = NotImplementedError()
    criterion = nn.CrossEntropyLoss()
    
    # ======================================================================
    # Define optimizer function.
    # ----------------- YOUR CODE HERE ----------------------
    #
    # Remove NotImplementedError and assign appropriate optimizer with learning rate and other paramters.
    # optimizer = NotImplementedError()
    optimizer = optim.SGD(model.parameters(), lr=0.5)
        
    
    # Create transformations to apply to each data sample 
    # Can specify variations such as image flip, color flip, random crop, ...
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    # Load datasets for training and testing
    # Inbuilt datasets available in torchvision (check documentation online)
    dataset1 = datasets.MNIST('./data/', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('./data/', train=False,
                       transform=transform)
    train_loader = DataLoader(dataset1, batch_size = 10, 
                                shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset2, batch_size = 10, 
                                shuffle=False, num_workers=4)
    
    best_accuracy = 0.0

    train_losses = []
    train_accuracies= []
    test_losses = []
    test_accuracies = []
    
    # Run training for n_epochs specified in config 
    for epoch in range(1, 5 + 1):
        import time
        print("Epoch start: {}".format(time.time()))
        train_loss, train_accuracy = train(model, device, train_loader,
                                            optimizer, criterion, epoch, 10)
