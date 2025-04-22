# Chest X-Ray Images (Pneumonia)
This repository tries to classify the chest X-ray images into two classes: pneumonia and normal. The dataset is taken from Kaggle and contains 3,875 images of pneumonia and 1,341 images of normal X-rays.

## Dataset
The dataset is available on Kaggle at the following link: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). The dataset is well structured and contains two folders: `train` and `test`. Each folder contains two subfolders: `NORMAL` and `PNEUMONIA`, which contain the respective images.
The dataset is divided into training and testing sets, with 3,875 images of pneumonia and 1,341 images of normal X-rays in the training set, and 390 images of pneumonia and 234 images of normal X-rays in the testing set.

## Requirements
All the requirements are in the `requirements.txt` file. You can install them using the following command:
```bash
pip install -r requirements.txt
```
## Usage
To run the code, you need to have the dataset in home direcotry of the proct if the dataset is not in the home directory, you need to change the path in the `config.py` file. The dataset should be in the following structure:
```
data/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
```

## Training
To train the model, run the following command:
```bash
python main.py
```
This will start the training of resnet model with the few twicking in the forward pass. The model will be saved in the `model` directory. The training process will take some time depending on the hardware you are using. The model will be trained for 10 epochs and the best model will be saved in the `model` directory.

## Testing
The `model.py` script will also automatically run the testing script once the model is fully trained. After the training is complete, the model will be tested on the test set and the accuracy will be printed on the console.

## Inference
During the testing phase, the model outputed the results as follows:
```
Accuracy: 0.8590
Precision: 0.8341
Recall: 0.9667
F1 Score: 0.8955
```

The script also outputs the two images representing the test and training loss during two different stages of the model training in the filename named as `first_train.png` and `second_train.png`. The first image represents the training loss during initial stages when the convolutional layers are frozen and classification layers are trained.

The second image represents the training loss during the final stages when the convolutional layers are unfrozen and the entire model is trained with the classification layers. The model is trained for 10 epochs and the best model is saved in the `model` directory.