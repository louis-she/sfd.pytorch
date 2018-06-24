import os

from torch.utils.data import Dataset

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

                file_path = lines[cursor]
                face_count = int(lines[cursor+1])
                bboxes = lines[cursor+2:cursor+2+face_count]

                coordinates = []
                for bbox in bboxes:
                    bbox = bbox.split(' ')
                    coordinate = (int(bbox[1]), int(bbox[0]), int(bbox[1])+int(bbox[3]), int(bbox[0])+int(bbox[2]))
                    coordinates.append(coordinate)

                processed_annotation.append({
                    'image_path': file_path,
                    'coordinates': coordinates
                })
                cursor = cursor + 2 + face_count


    return val_processed_annotation, train_processed_annotation


class FDDBDataset():

    def __init__(self, fddb_annotation_dir, fddb_dataset_dir):
        super().__init__()
        self.fddb_annotation_dir = fddb_annotation_dir
        self.fddb_dataset_dir = fddb_dataset_dir

    def __len__(self):
        return 10

    def __getitem__(self, index):
        return 10


if __name__ == "__main__":
    a, b = create_datasets('/Users/louis/Downloads/wider_face')
    print(a)