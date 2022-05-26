from torchvision.datasets import ImageFolder
import torch
import os
import collections


torch.manual_seed(0)

ImageItem = collections.namedtuple('ImageItem', ('image_name', 'tag'))

class RobustnessDataset(ImageFolder):
    def __init__(self, dataset_path):
        self._dataset_path = dataset_path
        self._tag_list = [tag for tag in os.listdir(self._dataset_path)]
        self._all_images = []
        for tag in self._tag_list:
            base_dir = os.path.join(self._dataset_path, tag)
            for i, file in enumerate(os.listdir(base_dir)):
                self._all_images.append(ImageItem(file, tag))

    def __getitem__(self, item):
        image_item = self._all_images[item]
        image_path = os.path.join(self._dataset_path, image_item.tag, image_item.image_name)
        return image_path

    def __len__(self):
        return len(self._all_images)