# Imports here
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import time
from PIL import Image
import copy
import json
import sys
import argparse
from prettytable import PrettyTable


# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'

    checkpoint = torch.load(filepath, map_location=map_location)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = checkpoint['optimizer']
    epochs = checkpoint['epochs']
    batch_size = checkpoint['batch_size']
    for param in model.parameters():
        param.requires_grad = False
        
    return model, checkpoint['class_to_idx']

def load_image(path, image_size = (224,224)):
    # TODO: Process a PIL image for use in a PyTorch model
    image = Image.open(path)
    preprocess = transforms.Compose([transforms.Resize(image_size),
                                                transforms.CenterCrop(image_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
                                                ])
    processed_image = preprocess(image)
    return image, processed_image


def predict(image, model,device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = image.unsqueeze(0)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    inputs = image.to(device)
    logits = model.forward(inputs)
    
    ps = F.softmax(logits,dim=1)
    topk = ps.cpu().topk(topk)
    probs, classes =  (e.data.numpy().squeeze().tolist() for e in topk)
    classes = [class_ - 1 for class_ in classes]
    return probs, classes
        
def print_pred(classes, probs, cat_to_name):
    table = PrettyTable() 
    table.field_names = ["Class Name", "Probability"]
    class_names = [cat_to_name[str(c)] for c in classes]
    
    for i,n in zip(class_names, probs):
        table.add_row([i, '%.2f'%(n)])
    print(table)

def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")
    script_path = sys.path[0]
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_path", help="The path the image for inference", type=str)

    ap.add_argument("--model_path", help="Path to checkpoint of the model",type=str)
    
    ap.add_argument("--cat_path", help="Path to category mapping",default=sys.path[0] + '//' + 'cat_to_name.json', type=str)
    
    ap.add_argument("--device", help="GPU or CPU for training. (default: GPU if avalaible)", 
                    choices=("cpu", "gpu"), default="cpu")

    
    args = vars(ap.parse_args())
    
    
   
    if args["device"] == "gpu" and str(device) == "cuda":
        device = torch.device("cuda") 
    else:
        device = torch.device("cpu")
        
    print("Selected device is: " + str(device))   
    
    
    image_path = script_path  + '/' + args["image_path"]
    checkpoint_path = script_path +'/' +  args["model_path"]
    cat_path = args["cat_path"]
    with open(cat_path, 'r') as f:
        cat_to_name = json.load(f)
    
    model, class_to_idx = load_checkpoint(checkpoint_path)
    image, processed_image = load_image(image_path ) 
    probs, classes = predict(processed_image, model, device)  
    print_pred(classes, probs, cat_to_name)
if __name__ == '__main__':
    main()
    