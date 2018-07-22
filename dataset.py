import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from random import uniform
from torchvision import transforms

from config import Config
from imageaug import crop_square, random_horizontal_flip

def my_collate_fn(batch):
    batch = [y for x in batch for y in x]
    images = torch.stack(list(map(lambda x: torch.tensor(x[0]), batch)))
    coordinates = list(map(lambda x: x[1], batch))
    pathes = list(map(lambda x: x[2], batch))
    scale = np.array(list(map(lambda x: x[3], batch)))

    return images, coordinates, pathes, scale


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
        image_size=Config.IMAGE_SIZE,
        random_flip=False, random_crop=False,
        random_color_jitter=False)

    return train_dataset, validation_dataset


class FDDBDataset(Dataset):

    def __init__(self, images_dir, annotation, image_size=640,
                 random_crop=Config.RANDOM_CROP, random_flip=Config.RANDOM_FLIP,
                 random_color_jitter=Config.RANDOM_COLOR_JITTER):
        super().__init__()
        self.images_dir = images_dir
        self.annotation = annotation
        self.image_size = image_size
        self.random_crop = random_crop
        self.random_color_jitter = random_color_jitter
        self.random_flip = random_flip
        self.transform = None

        self.init_transforms()

    def init_transforms(self):
        transform = [ transforms.ToPILImage() ]
        if self.random_color_jitter:
            transform.append(transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2
            ))
        transform.append(transforms.ToTensor())
        self.transform = transforms.Compose(transform)

    def __image_loader(self, image_path):
        return cv2.imread(image_path)

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        file_path, coordinates = self.annotation[index]
        file_path = os.path.join(self.images_dir, file_path)
        image = self.__image_loader(file_path)
        image -= np.array([104,117,123], dtype=np.uint8)

        images = []
        coordinates_list = []
        if self.random_crop:
            ratio = uniform(Config.MIN_CROPPED_RATIO, Config.MAX_CROPPED_RATIO)
            for _ in range(Config.CROPPED_IMAGE_COUNT):
                cropped_image, new_coordinates = crop_square(
                    image, coordinates, ratio, Config.KEEP_AREA_THRESHOLD)
                images.append(cropped_image)
                coordinates_list.append(new_coordinates)
            if Config.KEEP_LARGEST_SQUARE:
                cropped_image, new_coordinates = crop_square(image, coordinates, 1)
                images.append(cropped_image)
                coordinates_list.append(new_coordinates)
            if Config.KEEP_ORIGINAL:
                images.append(image)
                coordinates_list.append(coordinates)
        else:
            images.append(image)
            coordinates_list.append(coordinates)

        if self.random_flip:
            for index, image in enumerate(images):
                images[index], coordinates_list[index] = \
                    random_horizontal_flip(image, coordinates_list[index])
        result = []
        for index, image in enumerate(images):
            coordinates = coordinates_list[index]

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
            result.append((image, coordinates, file_path, (1 / height_scale, 1 / width_scale)))

        return result
