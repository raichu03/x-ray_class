import os

DATA_DIR = "./data/chest_xray"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

BATCH_SIZE = 16
IMG_SIZE = 224
LR = 1e-4
EPOCHS = 10
DEVICE = "cuda"