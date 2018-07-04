import os

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


def my_collate_fn(batch):
    images = torch.stack(list(map(lambda x: torch.tensor(x[0]).float(), batch)))
    coordinates = list(map(lambda x: x[1], batch))
    pathes = list(map(lambda x: x[2], batch))

    return images, coordinates, pathes


def create_datasets(dataset_dir, skip_invalid=True, skip_occlusion=True,
                    skip_illumination=False, skip_heavy_blur=True):
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
                face_count = int(lines[cursor+1])
                bboxes = lines[cursor+2:cursor+2+face_count]

                coordinates = []
                for bbox in bboxes:
                    bbox = bbox.split(' ')
                    coordinate = (int(bbox[1]), int(bbox[0]), int(bbox[1])+int(bbox[3]), int(bbox[0])+int(bbox[2]))
                    coordinates.append(coordinate)

                processed_annotation.append((
                    file_path,
                    coordinates
                ))
                cursor = cursor + 2 + face_count

    train_dataset = FDDBDataset(os.path.join(dataset_dir, 'WIDER_train/images'), train_processed_annotation)
    validation_dataset = FDDBDataset(os.path.join(dataset_dir, 'WIDER_val/images'), val_processed_annotation)

    return train_dataset, validation_dataset


class FDDBDataset(Dataset):

    def __init__(self, images_dir, annotation, image_size=640, transform=None):
        super().__init__()
        self.images_dir = images_dir
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


if __name__ == "__main__":
    a, b = create_datasets('/home/louis/datasets/wider_face')
    print(a[6])