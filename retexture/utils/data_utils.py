import os
import os.path as osp
from collections import Counter

import matplotlib.pyplot as plt
import requests
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

default_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class RetextureDataset(Dataset):
    def __init__(self, root_dir, transform=default_transform):
        """Custom PyTorch dataset for ReTexture.

        Args:
            root_dir (str): Root directory of the dataset.
            transform (callable, optional): A function/transform to apply to the images.
        """

        self.root_dir = root_dir
        self.transform = transform
        self.classes = [
            d
            for d in os.listdir(root_dir)
            if osp.isdir(osp.join(root_dir, d)) and not d.startswith(".")
        ]

        # Create a mapping from class names to class indices.
        self.class2idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.idx2class = {v: k for k, v in self.class2idx.items()}

        # Initialize a list to store image file paths and labels.
        self.samples = []

        self.traverse_files()
        self.mk_imagenet_map()

    @classmethod
    def _get_contents(self, name):
        if osp.isdir(name):
            files = os.listdir(name)
            return {f: self._get_contents(osp.join(name, f)) for f in files}
        else:
            return name

    def get_dataset(self):
        return self._get_contents(self.root_dir)

    def traverse_files(self):
        """deprecated in favor of get_dataset"""

        for class_name in self.classes:
            class_dir = osp.join(self.root_dir, class_name)

            if osp.isdir(class_dir):
                # Get the list of texture directories inside the current class_dir.
                texture_dirs = [
                    d
                    for d in os.listdir(class_dir)
                    if osp.isdir(osp.join(class_dir, d))
                ]

                for texture_dir in texture_dirs:
                    texture_dir_path = osp.join(class_dir, texture_dir)

                    for filename in os.listdir(texture_dir_path):
                        img_path = osp.join(texture_dir_path, filename)
                        if img_path.endswith(".jpg") or img_path.endswith(".png"):
                            self.samples.append((img_path, self.class2idx[class_name]))

    def mk_imagenet_map(self):
        LABELS_URL = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"

        response = requests.get(LABELS_URL)
        class_idx = response.json()
        RetextureDataset.imagenet_classes = {
            int(key): value[1] for key, value in class_idx.items()
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image) if self.transform else image

        # not doing one-hot because of csv file...
        # label = F.one_hot(torch.tensor(label), num_classes=len(self.classes)).float()

        return {"image": image, "label": label, "path": img_path}
