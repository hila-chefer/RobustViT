import json
from torch.utils import data
from torchvision.datasets import ImageFolder
import torch
import os
from PIL import Image
import numpy as np
import argparse
from tqdm import tqdm
from munkres import Munkres
import multiprocessing
from multiprocessing import Process, Manager
import collections
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
import torchvision
import cv2
import random
torch.manual_seed(0)

SegItem = collections.namedtuple('SegItem', ('image_name', 'tag'))

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])

TRANSFORM_TRAIN = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    ])

TRANSFORM_EVAL = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
])

IMAGE_TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    normalize
])

MERGED_TAGS = {'n04356056', 'n04355933',
         'n04493381', 'n02808440',
         'n03642806', 'n03832673',
         'n04008634', 'n03773504',
         'n03887697', 'n15075141'}

TRAIN_PARTITION = "train"
VAL_PARTITION = "val"
LEGAL_PARTITIONS = {TRAIN_PARTITION, VAL_PARTITION}

# TRAIN_CLASSES = 500

class SegmentationDataset(ImageFolder):
    def __init__(self, seg_path, imagenet_path, partition=TRAIN_PARTITION, num_samples=2, train_classes=500
                 , imagenet_classes_path='imagenet_classes.json', seed=None):
        assert partition in LEGAL_PARTITIONS
        self._partition = partition
        self._seg_path = seg_path
        self._imagenet_path = imagenet_path
        with open(imagenet_classes_path, 'r') as f:
            self._imagenet_classes = json.load(f)
        self._tag_list = [tag for tag in os.listdir(self._seg_path) if tag not in MERGED_TAGS]
        if seed:
            print(f'Shuffling training classes with seed {seed}')
            random.seed(seed)
            random.shuffle(self._tag_list)
        if partition == TRAIN_PARTITION:
            # Skip merged tags as those cause a headache
            self._tag_list = self._tag_list[:train_classes]
        elif partition == VAL_PARTITION:
            # Skip merged tags as those cause a headache
            self._tag_list = self._tag_list[train_classes:]
        for tag in self._tag_list:
            assert tag in self._imagenet_classes
        self._all_segementations = []
        for tag in self._tag_list:
            base_dir = os.path.join(self._seg_path, tag)
            for i, seg in enumerate(os.listdir(base_dir)):
                if i >= num_samples:
                    break
                self._all_segementations.append(SegItem(seg.split('.')[0], tag))

    def __getitem__(self, item):
        seg_item = self._all_segementations[item]

        seg_path = os.path.join(self._seg_path, seg_item.tag, seg_item.image_name + ".png")
        image_path = os.path.join(self._imagenet_path, seg_item.tag, seg_item.image_name + ".JPEG")

        seg_map = Image.open(seg_path)
        image = Image.open(image_path)
        image = image.convert('RGB')

        seg_map = np.array(seg_map)
        seg_map = seg_map[:, :, 1] * 256 + seg_map[:, :, 0]

        assert len([cand for cand in np.unique(seg_map) if cand != 0 and cand != 1000]) == 1

        # Convert to binary seg maps
        seg_map[seg_map == 1000] = 0
        seg_map[seg_map != 0] = 1

        seg_map = torch.from_numpy(seg_map.astype(np.float32))

        # transforms - start
        seg_map = seg_map.reshape(1, seg_map.shape[-2], seg_map.shape[-1])

        if self._partition == VAL_PARTITION:
            image = TRANSFORM_EVAL(image)
            seg_map = TRANSFORM_EVAL(seg_map)

        elif self._partition == TRAIN_PARTITION:
            # Resize
            resize = transforms.Resize(size=(256, 256))
            image = resize(image)
            seg_map = resize(seg_map)

            # Random crop
            i, j, h, w = transforms.RandomCrop.get_params(
                image, output_size=(224, 224))
            image = TF.crop(image, i, j, h, w)
            seg_map = TF.crop(seg_map, i, j, h, w)

            # RandomHorizontalFlip
            if random.random() > 0.5:
                image = TF.hflip(image)
                seg_map = TF.hflip(seg_map)

        else:
            raise Exception(f"Unsupported partition type {self._partition}")

        # normalize original image and turn to tensor
        image_ten = IMAGE_TRANSFORMS(image)
        # transforms - end

        class_name = int(self._imagenet_classes[seg_item.tag])

        return seg_map, image_ten, class_name

    def __len__(self):
        return len(self._all_segementations)