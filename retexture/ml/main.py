import argparse
import os

import torch
import torch.hub
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

"""
1. get the model
2. get the data
  2a. we need a data loader that decides / abstracts away how we get the data
3. train on the data ... this is done since we are using pretrained models
4. evaluate the model on some data

model is neural network which is a mathematical function
output = f(input)

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
        self.classes = os.listdir(root_dir)

        # Create a mapping from class names to class indices.
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # Initialize a list to store image file paths and labels.
        self.samples = []

        # Traverse the directory structure to collect image file paths and labels.
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                # Get the list of texture directories inside the current class_dir.
                texture_dirs = [
                    d
                    for d in os.listdir(class_dir)
                    if os.path.isdir(os.path.join(class_dir, d))
                ]

                for texture_dir in texture_dirs:
                    texture_dir_path = os.path.join(class_dir, texture_dir)
                    for filename in os.listdir(texture_dir_path):
                        img_path = os.path.join(texture_dir_path, filename)
                        if img_path.endswith(".jpg") or img_path.endswith(".png"):
                            # Store the image path and its corresponding class index.
                            self.samples.append(
                                (img_path, self.class_to_idx[class_name])
                            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = Image.open(img_path)
        image = self.transform(image) if self.transform else image

        label = F.one_hot(torch.tensor(label), num_classes=len(self.classes)).float()

        return image, label


def evaluate(model, dataloader):
    """Evaluate a model on a dataset and print logits to the command line.

    Args:
        model (torch.nn.Module): The pre-trained model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
    """

    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            logits = model(images)
            print("Logits:", logits)
            print("Label:", labels)


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
