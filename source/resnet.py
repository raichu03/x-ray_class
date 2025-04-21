import torch.nn as nn
from torchvision import models


def resnet_50(pretrained=True, freeze=True):
    model = models.resnet50(pretrained=pretrained)
    num_features = model.fc.in_features
    
    model.fc = nn.Sequential(
        nn.Linear(num_features, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
            
        nn.Linear(64, 1),
        nn.Sigmoid()
    )
    
    if freeze:
        for parm in model.parameters():
            parm.requires_grad = False
        for parm in model.fc.parameters():
            parm.requires_grad = True
    
    return model