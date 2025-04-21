import torch
from tqdm import tqdm
from .config import DEVICE

def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0
    
    for imgs, labels in tqdm(dataloader):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
        optimizer.zero_grad()
        outpus = model(imgs)
        loss = criterion(outpus, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    return running_loss / len(dataloader)

def validate(model, dataloader, criterion):
    model.eval()
    val_loss = 0
    
    for imgs, labels in dataloader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
    
    return val_loss / len(dataloader)

def unfreeze_last_block(model):
    for name, parm in model.named_parameters():
        if "layer4" in name or "fc" in name:
            parm.requires_grad = True