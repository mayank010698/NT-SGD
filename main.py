import argparse
import numpy as np
import json
from torchvision.io import read_image
import os
import torch
import torchvision
from torchvision import transforms
import torchmetrics
from datetime import datetime
from torchvision import models, datasets
from torch.utils.data import Subset,DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from matplotlib import pyplot as plt
import sys
from vgg_pytorch.model import vgg5
from truncate import NormTruncate, ETC
from utils import *
import pickle



device = "cuda" if torch.cuda.is_available() else "cpu"

step = 0

def train(epoch,  model ,optimizer , loss_fn, train_metric,
           test_metric, training_loader, test_loader, args, sw):

    global step

    train_loss      = 0
    test_loss       = 0
    train_batches   = 0
    test_batches    = 0

    #explore-then-commit truncation
    trunc             = ETC(model, args.p,args.thres)
    #Truncated-SGD
    # trunc             = NormTruncate(model,args.eps2)

    
    #Set model to train
    model.train()
    if(args.model_name=="resnet18"):
        model.apply(deactivate_batchnorm)
    
    train_metric.reset()
    test_metric.reset()

    for i, data in enumerate(training_loader):
        inputs, labels  = data
        inputs          = inputs.to(device)
        labels          = labels.to(device)

        optimizer.zero_grad()
        outputs         = model(inputs)
        train_batch_acc = train_metric(outputs,labels)
        sw.add_scalar('Training/TrainAccuracy',train_batch_acc.item(),step)
        loss            = loss_fn(outputs, labels)

        loss.backward()
        sparsity = trunc.step(model)

        optimizer.step()
        train_batches    = train_batches + 1
        train_loss += loss.item()

        sw.add_scalar('Training/Loss',loss.item(),step)
        for k,v in sparsity.items():
            sw.add_scalar('Training/Sparsity_Layer:{}'.format(k),v.item(),step)

        step +=1

        if(i%100==0):
            print("Epoch {}, Batch {}, Loss:{}".format(epoch, i, loss.item()))
        
        
    trunc.reset()
    model.eval()
    train_acc = train_metric.compute() 
    test_batches = 0

    for i, vdata in enumerate(test_loader):
        tinputs, tlabels = vdata
        tinputs     = tinputs.to(device)
        tlabels     = tlabels.to(device)
        toutputs    = model(tinputs)
        tloss       = loss_fn(toutputs, tlabels)
        test_loss  += tloss.item()
        test_batches = test_batches + 1
        
        test_batch_acc = test_metric(toutputs, tlabels)

    test_acc = test_metric.compute()
    
    return train_loss/train_batches, test_loss/test_batches, train_acc.cpu().numpy(),test_acc.cpu().numpy()

def train_and_plot():

    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()

    with open(args.config,"r") as f:
        json_file = json.load(f)

    #read config
    params = Config(json_file)
    if(params.dataset=="mnist" or params.dataset=="cifar10"):
        nc = 10
    else:
        nc = 100
    
    setattr(params,"nc",nc)
    params.run_comment = "{}_{}_{:.2f}".format(params.dataset,params.thres,params.p)

    sw = SummaryWriter(comment=params.run_comment)

    if not os.path.exists("models_"+params.run_comment):
        os.makedirs("models_"+params.run_comment)


    
    model           = load_model(params.model_name, params.dataset, params.nc).to(device)
    optimizer       = torch.optim.SGD(model.parameters(), lr=params.lr )
    loss_fn         = torch.nn.CrossEntropyLoss()
    train_metric    = torchmetrics.Accuracy(task="multiclass", num_classes=nc).to(device)
    test_metric     = torchmetrics.Accuracy(task="multiclass", num_classes=nc).to(device)
    train_loss      = []
    test_loss       = []
    train_acc       = []
    test_acc        = []

    
    train_dataset, test_dataset = load_data(params.model_name, params.dataset)
    train_loader, test_loader    = load_dataloader(train_dataset, test_dataset, params.batch_size)

    results = {}
    for epoch in range(params.epochs):

        trloss, tloss, tr_acc,t_acc = train(epoch, model, optimizer, loss_fn, train_metric,
                                             test_metric,  train_loader, test_loader, params , sw)

        print('Epoch :{}, Training Loss {} Validation Loss {}'.format(epoch, trloss, tloss))

        sw.add_scalar('Epochs/Training Loss',trloss, epoch)
        sw.add_scalar('Epochs/Test Loss',tloss, epoch)
        sw.add_scalar('Accuracy/Test', t_acc, epoch)
        sw.add_scalar('Accuracy/Train', tr_acc, epoch)

        train_loss.append(trloss)
        test_loss.append(tloss)
        train_acc.append(tr_acc)
        test_acc.append(t_acc)
        
        epoch  += 1


    results["training_loss"] = train_loss
    results["test_loss"] =  test_loss
    results["train_acc"] = train_acc
    results["test_acc"] = test_acc

    with open("{}_{}_{}_{}_NTSGD.json".format(params.model_name,params.dataset,params.thres,params.p),"wb") as f:
        pickle.dump(results,f)
    

if __name__=="__main__":
   train_and_plot()
