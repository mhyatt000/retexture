import argparse
from tqdm import tqdm
import torch.nn.functional as F
import pandas as pd
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

def get_args():
    parser = argparse.ArgumentParser(description="Parse command line arguments.")
    parser.add_argument(
        "--model_name", type=str, help="Specify the model name as a string."
    )
    parser.add_argument(
        "--root_data", type=str, help="Specify the root data directory as a string.", 
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
    def __init__(self, dataset):
        self.df = None  
        self.dataset = dataset

    def remember(self, gt, out, path):
        """
        gt is groud truth
        out is the output of the model
        """
        out = out.numpy().tolist()  
        data = {'path': path, 'gt': int(gt)}
        data.update({f'out{i}': out[i] for i in range(len(out))})
        
        new_row = pd.DataFrame([data])
        
        if self.df is None:
            self.df = new_row
        else:
            self.df = pd.concat([self.df, new_row], ignore_index=True)
       
    def show(self):
        print(self.df)

    def to_csv(self, path):
        self.df.to_csv(path, index=False)

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


def evaluate(model, dataloader, args):
    """Evaluate a model on a dataset and print logits to the command line.

    Args:
        model (torch.nn.Module): The pre-trained model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
    """

    A = Analyzer(dataloader.dataset)
    import time

    model.eval()
    with torch.no_grad():
        for batch  in tqdm(dataloader, total=args.batches):
            logits = model(batch['image'])
            for out, gt, path in zip(logits, batch['label'], batch['path']):
                A.remember(gt, F.softmax(out, dim=0), path)

    A.show()
    A.to_csv(f'{args.model_name}.csv')

def main():
    args = get_args()
    model = get_model(args.model_name)

    blender_dataset = RTD(root_dir=args.root_data)
    batch_size = 1
    args.batches = len(blender_dataset) // batch_size
    dataloader = DataLoader(blender_dataset, batch_size=batch_size, shuffle=True)

    evaluate(model, dataloader, args=args)


if __name__ == "__main__":
    main()
