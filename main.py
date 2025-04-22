from source.dataloader import get_dataloaders
from source.resnet import resnet_50
from source.train import train_one_epoch, validate, unfreeze_last_block
from source.evaluate import evaluate_model
from source.utils import plot_curves
from source.config import *

import torch
import torch.nn as nn
import torch.optim as optim


def main():
    train_loader, val_loader, test_loader = get_dataloaders()
    model = resnet_50(pretrained=True, freeze=True).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LR)
    
    train_losses, val_losses = [], []
    
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss = validate(model, val_loader, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    
    plot_curves(train_losses, val_losses, filename="first_train")
    
    for epoch in range(EPOCHS // 2):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss = validate(model, val_loader, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    plot_curves(train_losses, val_losses, filename="second_train")
    
    MODEL_PATH = "model/resnet50_pneumonia.pt"
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    
    acc, prec, rec, f1, cm = evaluate_model(model, test_loader)
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1-Score: {f1:.4f}")
    print("Confusion Matrix:\n", cm)


if __name__ == "__main__":
    main()