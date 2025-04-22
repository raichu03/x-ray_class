import matplotlib.pyplot as plt

def plot_curves(train_losses, val_losses, filename="loss_curve.png"):
    plt.plot(train_losses, label = "Train")
    plt.plot(val_losses, label = "Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.savefig(filename)
    plt.close()