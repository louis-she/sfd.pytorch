import torch
import cv2
from dataset import create_datasets
from model import Net

if __name__ == "__main__":

    train_dataset, val_dataset = create_datasets('/home/louis/datasets/wider_face')

    image, annotation = train_dataset[0]
    image = cv2.resize(image, (640, 640))

    _input = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float()

    net = Net()
    ret = net(_input)