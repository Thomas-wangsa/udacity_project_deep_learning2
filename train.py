import argparse

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder
import torch.nn.functional as F

from PIL import Image

from collections import OrderedDict

import time

import numpy as np
import matplotlib.pyplot as plt


def set_data(path,type_transform) :
    result = datasets.ImageFolder(path, transform=type_transform)
    return result


def set_loader(data,batch_size):
    result = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    return result

def save_checkpoint(path, model, optimizer, args, classifier,epochs):
    
    if args.arch == "vgg16" :
        input_size = model.classifier[0].in_features
    elif args.arch == "densenet121" :
        input_size = model.classifier.in_features
    else :
        print("generate default input size")
        input_size = model.classifier[0].in_features

    checkpoint = {'input_size': input_size,
                  'output_size': 102,
                  'arch': args.arch,
                  'classifier' : classifier,
                  'learning_rate': args.learning_rate,
                  'epochs': epochs,
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                 }

    torch.save(checkpoint, path)

def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('--data_dir', action='store', default='flowers')
    parser.add_argument('--arch', dest='arch', default='vgg16')
    parser.add_argument('--learning_rate', dest='learning_rate', default='0.001')
    parser.add_argument('--hidden_units', dest='hidden_units', default='500')
    parser.add_argument('--epochs', dest='epochs', default='3')
    parser.add_argument('--gpu', action="store_true", default="false")
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="checkpoint2.pth")
    return parser.parse_args()

def train(model, criterion, optimizer, trainloader,vloader, epochs, gpu):
    steps = 0
    print_every = 10
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader): # 0 = train
            steps += 1 
            #if torch.cuda.is_available(): # testing this out, uncomment later
               # model.cuda()
            if gpu == True :
                model.cuda()
                inputs, labels = inputs.to('cuda'), labels.to('cuda') # use cuda

            #inputs, labels = inputs.to('cuda'), labels.to('cuda') # use cuda # uncomment later
            optimizer.zero_grad()
            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                valloss = 0
                accuracy=0

                for ii, (inputs2,labels2) in enumerate(vloader): # 1 = validation 
                        optimizer.zero_grad()
                        
                        if gpu == True :
                            inputs2, labels2 = inputs2.to('cuda') , labels2.to('cuda') # use cuda
                            model.to('cuda:0') # use cuda
                        
                        #if torch.cuda.is_available(): # commenting out to work with later possibly 
                            #inputs2, labels2 = inputs2.to('cuda') , labels2.to('cuda') # use cuda
                            #model.to('cuda:0') # use cuda
                        with torch.no_grad():    
                            outputs = model.forward(inputs2)
                            valloss = criterion(outputs,labels2)
                            ps = torch.exp(outputs).data
                            equality = (labels2.data == ps.max(1)[1])
                            accuracy += equality.type_as(torch.FloatTensor()).mean()

                valloss = valloss / len(vloader)
                accuracy = accuracy /len(vloader)

                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Training Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Loss {:.4f}".format(valloss),
                      "Accuracy: {:.4f}".format(accuracy),
                     )

                running_loss = 0
            
def main():
    print("train.py start") 
    args = parse_args()
    
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    means = [0.485, 0.456, 0.406]
    standard_deviations = [0.229, 0.224, 0.225]
    
    train_transforms = transforms.Compose([
                                        transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(means,standard_deviations)
                                    ])


    test_validation_struct = transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(means,standard_deviations)
                                    ])

    validation_transform = test_validation_struct
    test_transform = test_validation_struct
    
    print("set dataset start") 
    train_data = set_data(train_dir,train_transforms)
    validation_data = set_data(valid_dir,validation_transform)
    test_data = set_data(test_dir,test_transform)
    print("set dataset done")
    
    print("set loader start") 
    trainloader = set_loader(train_data,64)
    vloader = set_loader(validation_data,32)
    testloader = set_loader(test_data,20)
    print("set loader done") 
   
    model = getattr(models, args.arch)(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False
        
        
        
        
    if args.arch == "vgg16" :
        input_size = model.classifier[0].in_features
    elif args.arch == "densenet121" :
        input_size = model.classifier.in_features
    else :
        print("generate default input size")
        input_size = model.classifier[0].in_features
        
    print("set classifier start")
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, args.hidden_units)),
        ('drop', nn.Dropout(p=0.6)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(args.hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))]))
    
    model.classifier = classifier
    criterion = nn.NLLLoss() # using criterion and optimizer similar to pytorch lectures (densenet)
    optimizer = optim.Adam(model.classifier.parameters(), lr=float(args.learning_rate))
    epochs = int(args.epochs)
    print("SET epochs : {}".format(epochs))
    
    class_index = train_data.class_to_idx
    gpu = args.gpu # get the gpu settings
    print("SET GPU : {}".format(gpu))
    
    print("set train start")
    train(model, criterion, optimizer, trainloader,vloader, epochs, gpu)
    model.class_to_idx = class_index
    path = args.save_dir # get the new save location
    print("set save_checkpoint start")
    save_checkpoint(path, model, optimizer, args, classifier,epochs)


if __name__ == "__main__":
    main()