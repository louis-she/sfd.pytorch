import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.cross_validation import train_test_split
from config import Config

import xml.etree.ElementTree as ET


def my_collate_fn(batch):

    images = torch.stack(list(map(lambda x: torch.tensor(x[0]).double(), batch)))
    coordinates = list(map(lambda x: x[1], batch))
    pathes = list(map(lambda x: x[2], batch))

    return images, coordinates, pathes


def create_voc_datasets(voc_dataset_dir, split_ratio=0.2):
    annotations_dir = os.path.join(voc_dataset_dir, 'Annotations')
    data_dir = os.path.join(voc_dataset_dir, 'JPEGImages')
    imgs = os.listdir(data_dir)

    coordinates = []

    for img in imgs:
        annotation = img.split('.')[0] + '.xml'
        annotation_path = os.path.join(annotations_dir, annotation)


        def extract_info(annotation_path, classes=Config.VOC_CLASS):
            tree = ET.parse(annotation_path)
            root = tree.getroot()
            file_path = root.findall('filename')[0].text
            coordinates = []

            for i in root.findall('object'):
                if i.find('name').text == classes:
                    xmin = i.find('bndbox').find('xmin').text
                    ymin = i.find('bndbox').find('ymin').text
                    xmax = i.find('bndbox').find('xmax').text
                    ymax = i.find('bndbox').find('ymax').text

                    try:
                        coordinate = (int(xmin), int(ymin), int(xmax), int(ymax))
                        coordinates.append(coordinate)
                    except:
                        continue
                else:
                    continue
            return (file_path, coordinates)

        item = extract_info(annotation_path)
        if len(item[1]) == 0:
            continue
        else:
            coordinates.append(item)

    train_annotation, val_annotation = train_test_split(coordinates, test_size=0.2)

    train_dataset = VOCDataset(
        os.path.join(voc_dataset_dir),
        train_annotation,
        image_size=Config.IMAGE_SIZE)

    validation_dataset = VOCDataset(
        os.path.join(voc_dataset_dir),
        val_annotation,
        image_size=Config.IMAGE_SIZE)

    return train_dataset, validation_dataset


class VOCDataset(Dataset):

    def __init__(self, images_dir, annotation, image_size=640, transform=None):
        super().__init__()
        self.images_dir = os.path.join(images_dir, 'JPEGImages')
        self.annotation = annotation
        self.transform = transform
        self.image_size = image_size

    def __image_loader(self, image_path):
        return cv2.imread(image_path)

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        file_path, coordinates = self.annotation[index]
        file_path = os.path.join(self.images_dir, file_path)
        image = self.__image_loader(file_path)

        # scale coordinate
        height, width = image.shape[:2]
        width_scale, height_scale = 640.0 / width, 640.0 / height
        coordinates = np.array(list(map(lambda x: [
            x[0] * height_scale,
            x[1] * width_scale,
            x[2] * height_scale,
            x[3] * width_scale
        ], coordinates)))
        image = cv2.resize(image, (self.image_size, self.image_size))

        if self.transform:
            image = self.transform(image)

        return (image, coordinates, file_path)
