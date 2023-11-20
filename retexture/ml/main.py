import argparse
import os
import os.path as osp
from collections import Counter

import matplotlib.pyplot as plt
import requests
import torch
import torch.hub
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

"""
NOTES
top5 acc

how much did it give to shape? and to texture?
we care more about biological categories
- hard to find specific 3d models

TODO
1. find a way to consolidate / expand categories for 1-1 mapping
2. find top5
3. ask did any model or textures or NONE get classified?
    4. if NONE, compare probability of texture and object

WARNINGS:
- tested ≈ 10 images ... 2 or less had models or textues in top5 
- reviewers might wonder if the model is confused because of texture or because of distribution shift from rendering
    - solution... ?
        - control for the effect of rendering images
        - by rendering black texture "shillouette"
"""


def get_args():
    parser = argparse.ArgumentParser(description="Parse command line arguments.")
    parser.add_argument(
        "--model_name", type=str, help="Specify the model name as a string."
    )
    parser.add_argument(
        "--root_data", type=str, help="Specify the root data directory as a string."
    )

    return parser.parse_args()


def get_model(model_name):
    """Get a pre-trained model from PyTorch Hub.

    Args: model_name (str): Name of the model to retrieve. Options are "resnet", "vit", "swin", or "clip".
    Returns: torch.nn.Module: Pre-trained model.

    TODO:
    - when running CLIP provide class captions of the classes of interest
    - also use shape biased ResNet (refer to paper: https://arxiv.org/pdf/1811.12231.pdf)
        - last of 3 models... trained on imagenet and stylized imagenet
    """

    model_loaders = {
        "resnet": lambda: torch.hub.load(
            "pytorch/vision:v0.10.0", "resnet50", pretrained=True
        ),
        "vit": lambda: torch.hub.load(
            "facebookresearch/deit:main", "deit_base_patch16_224", pretrained=True
        ),
        "swin": lambda: torch.hub.load(
            "zihangJiang/transformer:main",
            "swin_base_patch4_window7_224",
            pretrained=True,
        ),
        "clip": lambda: torch.hub.load("openai/clip:v100", "ViT-B/32", pretrained=True),
    }

    if model_name in model_loaders:
        return model_loaders[model_name]()
    else:
        raise ValueError(
            "Unsupported model name. Please choose from 'resnet', 'vit', 'swin', or 'clip'."
        )

# vector rank=1000 [ , , , , , ... , n=1000]

class BlenderDS(Dataset):
    def __init__(self, root_dir, transform=None):
        """Custom PyTorch dataset for BlenderDS.

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

    def traverse_files(self):
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
        BlenderDS.imagenet_classes = {
            int(key): value[1] for key, value in class_idx.items()
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image) if self.transform else image

        label = F.one_hot(torch.tensor(label), num_classes=len(self.classes)).float()

        return image, label


class Analyzer:
    def __init__(self, dataset: BlenderDS):
        self.srcs = None
        self.tgts = None
        self.dataset = dataset
        self.tops = {}

    def _concat_tensor(self, buffer, tensor):
        """Private helper method to concatenate tensors."""
        return tensor if buffer is None else torch.cat((buffer, tensor), dim=0)

    def remember(self, src, tgt):
        self.srcs = self._concat_tensor(self.srcs, src)
        self.tgts = self._concat_tensor(self.tgts, tgt)

    def process(self):
        if self.srcs is None or self.tgts is None:
            print("Buffers are empty or incomplete.")
            return

        for i in range(self.tgts.shape[1]):  # Iterate over each column
            idxs = (self.tgts[:, i] == 1).nonzero(as_tuple=True)[0]
            selected_srcs = self.srcs[idxs]

            # Using argmax to get the top 5 for each selected src
            top5_values, top5_indices = torch.topk(selected_srcs, 5, dim=1)

            # Map indices to ImageNet classes
            top5_classes = sum(
                [
                    [self.dataset.imagenet_classes[idx.item()] for idx in indices]
                    for indices in top5_indices
                ],
                [],
            )

            self.tops[self.dataset.idx2class[i]] = top5_classes

    def visualize(self):
        if not hasattr(self, "tops") or not self.tops:
            print("No data to visualize.")
            return

        for k, v in self.tops.items():
            # Count the frequency of each class in v
            class_counts = Counter(v)

            # Prepare data for the histogram
            labels, values = zip(*class_counts.items())

            # Create horizontal bar plot
            plt.figure(figsize=(10, 5))
            plt.barh(labels, values, align="center")
            plt.xlabel("Frequency")
            plt.title(f"Histogram for {k}")

            # Show the plot
            plt.show()


def evaluate(model, dataloader):
    """Evaluate a model on a dataset and print logits to the command line.

    1. get the model
    2. get the data
      2a. we need a data loader that decides / abstracts away how we get the data
    3. train on the data ... this is done since we are using pretrained models
    4. evaluate the model on some data

    model is neural network which is a mathematical function
    output = f(input)

    Args:
        model (torch.nn.Module): The pre-trained model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
    """

    A = Analyzer(dataloader.dataset)
    i = 0

    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            logits = model(images)
            A.remember(src=logits, tgt=labels)

            # print("Logits:", logits)
            # print(logits.shape)
            # print("Label:", labels)

            i += 1
            if i == 2:
                A.process()
                A.visualize()
                quit()


def main():
    args = get_args()
    model = get_model(args.model_name)

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    blender_dataset = BlenderDS(root_dir=args.root_data, transform=transform)

    batch_size = 64
    dataloader = DataLoader(blender_dataset, batch_size=batch_size, shuffle=True)

    evaluate(model, dataloader)


if __name__ == "__main__":
    main()
