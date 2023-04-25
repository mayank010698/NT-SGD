
from torchvision.io import read_image
import torch
import torchvision
from torchvision import transforms
import torchmetrics
from torchvision import models, datasets
from torch.utils.data import Subset,DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from matplotlib import pyplot as plt
from vgg_pytorch.model import vgg5


def load_model(model_name,dataset,num_classes):
    if model_name=="resnet18":
        model = models.resnet18()
        if dataset == "mnist":
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        linear_size = list(model.children())[-1].in_features
        model.fc = nn.Linear(linear_size, num_classes)
    elif model_name == "vgg5":
        model = vgg5(num_classes=num_classes)
        if dataset == "mnist":
            model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        linear_size = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(linear_size, num_classes)

    else:
        raise NotImplementedError()
    
    return model

def load_data(model="resnet18",data="mnist"):
    T = [transforms.ToTensor()]

    if data == "mnist":
        T.append(transforms.Normalize((0.1307,), (0.3081,)))
        T = transforms.Compose(T)
        
        mnist_train = datasets.MNIST(root='../mnist',train=True,transform=T,download=True)

        digit_indices = {digit: [] for digit in range(10)}

        for index, (image, label) in enumerate(mnist_train):
            digit_indices[label].append(index)

        subset_indices = []
        for digit in range(10):
            subset_indices += digit_indices[digit][:1000]

        train_dataset = Subset(mnist_train, subset_indices)
        test_dataset = datasets.MNIST(root='../mnist',train=False,transform=T)

    elif data == "cifar10":
        img_transforms  = [transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip()]
        img_transforms.extend(T)
        img_transforms.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        img_transforms = transforms.Compose(img_transforms)
        train_dataset = datasets.CIFAR10(root='../cifar10',train=True,transform=img_transforms,download=True)
        test_dataset = datasets.CIFAR10(root='../cifar10',train=False,transform=img_transforms,download=True)
    elif data == "cifar100":
        img_transforms  = [transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip()]
        img_transforms.extend(T)
        img_transforms.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        img_transforms = transforms.Compose(img_transforms)
        train_dataset = datasets.CIFAR100(root='../cifar100',train=True,transform=img_transforms,download=True)
        test_dataset = datasets.CIFAR100(root='../cifar100',train=False,transform=img_transforms,download=True)
    else:
        raise NotImplementedError()
    
    return train_dataset, test_dataset 

def load_dataloader(train_dataset, test_dataset,batch_size=100):
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, test_loader

def deactivate_batchnorm(m):
    if isinstance(m, nn.BatchNorm2d):
        m.reset_parameters()
        m.eval()
        with torch.no_grad():
            m.weight.fill_(1.0)
            m.bias.zero_()

class Config:
    def __init__(self, json_file):
        for k, v in json_file.items():
            setattr(self, k, v)