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

from retexture.utils.data_utils import RetextureDataset as RTD

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


class Analyzer:
    def __init__(self, dataset: RTD):
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
            try:
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
            except:
                pass


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

            print("Logits:", logits)
            print(logits.shape)
            print("Label:", labels)

            i += 1
            if i == 2:
                A.process()
                A.visualize()
                quit()


def main():
    args = get_args()
    model = get_model(args.model_name)

    blender_dataset = RTD(root_dir=args.root_data)
    batch_size = 64
    dataloader = DataLoader(blender_dataset, batch_size=batch_size, shuffle=True)

    evaluate(model, dataloader)


if __name__ == "__main__":
    main()
