import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from random import uniform
from torchvision import transforms
from random import random

from config import Config
from imageaug import crop_square, random_horizontal_flip

def my_collate_fn(batch):
    images = torch.stack(list(map(lambda x: torch.tensor(x[0]), batch)))
    coordinates = list(map(lambda x: x[1], batch))
    pathes = list(map(lambda x: x[2], batch))

    return images, coordinates, pathes


def create_wf_datasets(dataset_dir):
    annotations_dir = os.path.join(dataset_dir, 'wider_face_split')
    val_annotation = os.path.join(annotations_dir, 'wider_face_val_bbx_gt.txt')
    train_annotation = os.path.join(annotations_dir, 'wider_face_train_bbx_gt.txt')

    val_processed_annotation = []
    train_processed_annotation = []

    for mode in ['train', 'val']:
        if mode == 'train':
            annotation_file = train_annotation
            processed_annotation = train_processed_annotation
        else:
            annotation_file = val_annotation
            processed_annotation = val_processed_annotation

        with open(annotation_file) as f:
            lines = f.readlines()
            cursor = 0

            while True:
                if len(lines) == cursor:
                    break

                file_path = lines[cursor][:-1]
                face_count = int(lines[cursor + 1])
                bboxes = lines[cursor + 2:cursor + face_count + 2]

                coordinates = []
                for bbox in bboxes:
                    bbox = bbox.split(' ')
                    if int(bbox[7]) == 1:
                        continue
                    coordinate = (
                        int(bbox[1]), int(bbox[0]),
                        int(bbox[1]) + int(bbox[3]),
                        int(bbox[0]) + int(bbox[2]),
                        1)  # one represents the class of face
                    coordinates.append(coordinate)

                processed_annotation.append((
                    file_path, coordinates))
                cursor = cursor + 2 + face_count

    train_dataset = FDDBDataset(
        os.path.join(dataset_dir, 'WIDER_train/images'),
        train_processed_annotation,
        image_size=Config.IMAGE_SIZE,
        random_color_jitter=Config.RANDOM_COLOR_JITTER)

    validation_dataset = FDDBDataset(
        os.path.join(dataset_dir, 'WIDER_val/images'),
        val_processed_annotation,
        image_size=Config.IMAGE_SIZE, mode='val')

    return train_dataset, validation_dataset


class FDDBDataset(Dataset):

    def __init__(self, images_dir, annotation, image_size=640,
                 random_flip=Config.RANDOM_FLIP, random_crop=True,
                 random_color_jitter=Config.RANDOM_COLOR_JITTER,
                 mode='train'):
        super().__init__()
        self.images_dir = images_dir
        self.annotation = annotation
        self.image_size = image_size
        self.random_color_jitter = random_color_jitter
        self.random_flip = random_flip
        self.transform = None
        self.random_crop = random_crop
        self.mode = mode

        # self.init_transforms()

    # def init_transforms(self):
    #     transform = [ transforms.ToPILImage() ]
    #     if self.random_color_jitter:
    #         transform.append(transforms.ColorJitter(
    #             brightness=0.2,
    #             contrast=0.2,
    #             saturation=0.2
    #         ))
    #     transform.append(transforms.ToTensor())
    #     self.transform = transforms.Compose(transform)

    def __image_loader(self, image_path):
        return cv2.imread(image_path)

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        file_path, coordinates = self.annotation[index]
        file_path = os.path.join(self.images_dir, file_path)
        image = self.__image_loader(file_path)
        image = image - np.array([104, 117, 123], dtype=np.uint8)

        if self.mode == 'train':
            if random() < 0.5:
                ratio = uniform(Config.MIN_CROPPED_RATIO, Config.MAX_CROPPED_RATIO)
            else:
                ratio = 1

            image, coordinates = crop_square(
                image, coordinates, ratio, Config.KEEP_AREA_THRESHOLD)
            image, coordinates = \
                random_horizontal_flip(image, coordinates)

            # scale coordinate
            height, width = image.shape[:2]
            width_scale, height_scale = 640.0 / width, 640.0 / height
            coordinates = np.array(list(map(lambda x: [
                x[0] * height_scale,
                x[1] * width_scale,
                x[2] * height_scale,
                x[3] * width_scale,
                *x[4:]
            ], coordinates)))

            image = cv2.resize(image, (self.image_size, self.image_size))
            if self.transform:
                image = self.transform(image)

        return image, coordinates, file_path
