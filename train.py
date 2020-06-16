import numpy as np
import matplotlib.pyplot as plt
import torch, gc
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import time
from PIL import Image
import copy
from collections import OrderedDict
import json  
import argparse
import os
import sys


def dataloader(data_dir,image_size = (224, 224),  batch_size = 32, ):
    """ Creates and returns dataloadeers and all relavent information"""

# TODO: Define your transforms for the training, validation, and testing sets
    
    data_train_transforms = transforms.Compose([transforms.Resize(image_size),
                                                transforms.RandomRotation(20),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.RandomVerticalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
                                                ])


    data_valid_transforms = transforms.Compose([transforms.Resize(image_size),
                                                transforms.CenterCrop(image_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
                                                ])

# TODO: Load the datasets with ImageFolder
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
   
    image_datasets = {}
    image_datasets['train'] = datasets.ImageFolder(train_dir, data_train_transforms)
    image_datasets['valid'] = datasets.ImageFolder(valid_dir, data_valid_transforms)            
    image_datasets['test'] = datasets.ImageFolder(test_dir, data_valid_transforms)
   

    # TODO: Using the image datasets and the trainforms, define the dataloaders

    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers = 0)
    dataloaders['valid'] = torch.utils.data.DataLoader(image_datasets['valid'], batch_size=batch_size, shuffle=True,num_workers = 0)
    dataloaders['test'] = torch.utils.data.DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False,num_workers = 0)
   
    class_names = image_datasets['train'].classes
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
    class_names = image_datasets['train'].classes
    return dataloaders,class_names,dataset_sizes, image_datasets,batch_size,image_size


# TODO: Build and train your network

def classifier_architecure(depth,classifier_parameters,dropout):
    """It's used to created a classifier of various depth"""
    classifier_depth_0 = nn.Sequential(OrderedDict([
                    ('fc1', nn.Linear(classifier_parameters[0], classifier_parameters[1])),
                    ('drop_out', nn.Dropout(p=dropout)),
                    ('output', nn.LogSoftmax(dim=1))]))
    classifier_depth_1 = nn.Sequential(OrderedDict([
                    ('fc1', nn.Linear(classifier_parameters[0], classifier_parameters[1])),
                    ('relu', nn.ReLU()),
                    ('drop_out', nn.Dropout(p=dropout)),
                    ('fc2', nn.Linear(classifier_parameters[1], classifier_parameters[2])),
                    ('output', nn.LogSoftmax(dim=1))]))
    classifier_depth_2 = nn.Sequential(OrderedDict([
                    ('fc1', nn.Linear(classifier_parameters[0], classifier_parameters[1])),
                    ('relu', nn.ReLU()),
                    ('drop_out', nn.Dropout(p=dropout)),
                    ('fc2', nn.Linear(classifier_parameters[1], classifier_parameters[2])),
                    ('relu', nn.ReLU()),
                    ('drop_out', nn.Dropout(p=dropout)),
                    ('fc3', nn.Linear(classifier_parameters[2], classifier_parameters[3])),
                    ('output', nn.LogSoftmax(dim=1))]))
        
    classifier_depth_list = [classifier_depth_0,classifier_depth_1,classifier_depth_2]
    classifier = classifier_depth_list[depth]
    return classifier

def classifier(model_name,device, learning_rate, dropout, depth):
    """ Creates a new model on a pretrained one with a replaced classifier"""
    criterion = nn.NLLLoss()
    model_list = [models.resnet18(pretrained=True),models.densenet161(pretrained=True)]
    model_name == "resnet18"
    if model_name == "resnet18":
        model = model_list[0]
        classifier_parameters_list = [[512, 102, 1, 1],[512, 128, 102, 1],[512, 256, 128, 102]]
        for param in model.parameters():
            param.requires_grad = False
            
        model.fc = classifier_architecure(depth,classifier_parameters_list[depth],dropout)
        optimizer = optim.SGD(model.fc.parameters(), lr=learning_rate)  
        
    elif model_name == "densenet161":
        model = model_list[1]
        classifier_parameters_list = [[2208, 102, 1, 1],[2208, 552, 102, 1],[2208, 1104, 552, 102]]
        for param in model.parameters():
            param.requires_grad = False
            
        model.classifier = classifier_architecure(depth,classifier_parameters_list[depth],dropout)
        optimizer = optim.SGD(model.classifier.parameters(), lr=learning_rate)  

        
    return model, criterion, optimizer



def train_model(model, dataloaders,stop_num, optimizer,epochs,device, criterion = nn.NLLLoss(), 
                train_losses_history =[], eval_losses_history =[],val_acc_history = [], 
                epoch_counts = 0):
    """ Train and eval phases are separate for clarity. Returns a trained model and various 
    statistics and parameters for vizualization and saving"""
    model.to(device)
    start = time.time()
    epoch_counts = 0
    best_acc = 0
    running_loss = 0
    bad_acc_epoch = 0
    
    for epoch in range(epochs):  
        if bad_acc_epoch < stop_num:
        
            epoch_counts += 1
            valid_size = 0
            train_size = 0
            running_corrects = 0
            train_epoch_loss = 0
            valid_epoch_loss = 0
            running_train_loss = 0
            running_valid_loss = 0
    
            for phase in ['train', 'valid']:
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    if phase == 'train':
                        model.train() 
                        # zero the parameter gradients
                        optimizer.zero_grad()
                        # forward
                        with torch.set_grad_enabled(True) :
                            outputs = model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)
                            loss.backward()
                            optimizer.step()
                            train_size +=1
                            running_train_loss += loss.item()
                    # eval phase 
                    if phase == "valid":
      
                        model.eval()
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        with torch.set_grad_enabled(False):
                            outputs = model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)
                            running_valid_loss += loss.item()
                        valid_size +=1
                        running_corrects += torch.sum(preds == labels.data)
        # print epoch statistics
    
            train_epoch_loss = running_train_loss/len(dataloaders["train"])
            valid_epoch_loss = running_valid_loss/len(dataloaders["valid"])
            epoch_acc = running_corrects.double() / len(dataloaders["valid"].dataset)
    
            print('Epoch : ',epoch+1, '\t', 'train loss :', train_epoch_loss,'\t', 'val loss :', valid_epoch_loss,
          'val acc :', epoch_acc.item())
     
        # copy the best model
            val_acc_history.append(epoch_acc)
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model.state_dict())
                bad_acc_epoch = 0
                best_train_losses_history = train_losses_history
                best_eval_losses_history = eval_losses_history
                best_epoch_counts = epoch_counts
            else:
                bad_acc_epoch +=1
            
            train_losses_history.append(train_epoch_loss)
            eval_losses_history.append(valid_epoch_loss)
            
        else:
            break        
    # general statistics
    print('Finished Training')
    model.load_state_dict(best_model)
    time_elapsed = time.time() - start
    print(" Traning complete in {:.0f}m {:.0f}s".format(time_elapsed//60, time_elapsed% 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return [model, best_train_losses_history, best_eval_losses_history,
            val_acc_history, best_epoch_counts]

# TODO: Do validation on the test set
def testing_model(test_loader, model, device):
    model.to(device)
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in test_loader:
            
            inputs, expected_labels = batch
            inputs, expected_labels = inputs.to(device), expected_labels.to(device)
            outputs =  model(inputs)

            predicted_labels = torch.argmax(outputs, dim=1)
            total += expected_labels.size(0)

            correct += (predicted_labels == expected_labels).sum().item()
    
    model_accuracy = 100 * correct / total
    return model_accuracy

# TODO: Save the checkpoint
# saving the model
def save_checkpoint(model,image_datasets,epoch_counts,class_names,optimizer, batch_size, criterion,script_path,model_name):
    model.class_to_idx = image_datasets['train'].class_to_idx
    checkpoint = {'output_size': len(class_names),
                  'epochs': epoch_counts,
                  'batch_size': batch_size,
                  'model': model,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'criterion':criterion
                 }
   
    torch.save(checkpoint, script_path + '//' + model_name + '.pth')
    print("Model saved at {}".format(str(script_path)))

def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")
    script_path = sys.path[0]

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", help="The dataset directory(default: 'flowers')",
                    default='flowers', type=str, nargs="?")

    VALID_ARCH_CHOICES = ('resnet18','densenet161' )
    ap.add_argument("--model", help="Pretrained models - resnet18 ; densenet161 (default: 'resnet18')", 
                                choices=VALID_ARCH_CHOICES, default=VALID_ARCH_CHOICES[0],type=str)
    
    VALID_DEPTH_CHOICES = (0, 1, 2)
    ap.add_argument("--depth", help="Depth of the classifier from 0 to 2. (default: 0)",
                    choices=VALID_DEPTH_CHOICES,  default=VALID_DEPTH_CHOICES[0], type=int)

    ap.add_argument("--learning_rate", help="Learning rate for SGD optimizer. (default: 0.01)",
                    default=0.01, type=float)
                    
    ap.add_argument("--dropout", help="Dropout P for the classifier. (default: 0.2)",
                    default=0.2, type=float)

    ap.add_argument("--epochs", help="Number of epochs. (default: 100)",
                    default=100, type=int)

    ap.add_argument("--stop_num", help="Early stop after validation accuracy decrease for n epochs. (default: 10)",
                    default=10, type=int)
    
    ap.add_argument("--device", help="GPU or CPU for training. (default: GPU if avalaible)", 
                    choices=("cpu", "gpu"), default="cpu")
    
    ap.add_argument("--model_name", help="Name for saved model. (default: checkpoint)", default="checkpoint", type=str)

    
    args = vars(ap.parse_args())
    
    
   
    if args["device"] == "gpu" and str(device) == "cuda":
        device = torch.device("cuda") 
    else:
        device = torch.device("cpu")
        
    print("Selected device is: " + str(device))    
   
    data_dir = script_path +"//" + args["data_dir"]
    model_name = args["model"]
    dropout = args["dropout"]
    depth = args["depth"]
    learning_rate = args["learning_rate"]
    epochs = args["epochs"]
    stop_num = args["stop_num"]
    checkpoint_name = args["model_name"]
    

    
    dataloaders,class_names,dataset_sizes,image_datasets,batch_size,image_size = dataloader(data_dir)
    model, criterion, optimizer = classifier(model_name,device, learning_rate, dropout, depth)
    model, train_losses_history, eval_losses_history, val_acc_history, epoch_counts = train_model(model,dataloaders,stop_num,optimizer, 
                                                                                                  epochs,device,criterion)
    test_accuracy = testing_model(dataloaders["test"], model,device)
    print('Test acc:  %.1f %%' % test_accuracy)
    save_checkpoint(model,image_datasets, epoch_counts,class_names,optimizer, batch_size, criterion, script_path, checkpoint_name)
    
if __name__ == '__main__':
    main()