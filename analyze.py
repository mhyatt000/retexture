import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import requests
import seaborn as sns


class DataFrameHeatMap:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self.mk_imagenet_map()

    def process_data(self):
        # Assuming 'gt' needs to be one-hot encoded. If it's already, adjust accordingly.
        gt_one_hot = pd.get_dummies(self.df["gt"], prefix="gt")

        # Aggregate 'out' columns by 'gt' values
        out_columns = [f"out{i}" for i in range(1000)]  # Adjust range if necessary
        self.df[out_columns] = self.df[out_columns]
        aggregated_data = self.df[out_columns].groupby(self.df["gt"]).sum()

        data = aggregated_data.to_numpy().T
        return data

    def get_paths_by_gt(self):
        paths_by_gt = {}
        for gt in self.df["gt"].unique():
            paths = self.df[self.df["gt"] == gt]["path"].tolist()
            paths_by_gt[gt] = paths[0].split("/")[-1].split("_")[0]
        return paths_by_gt

    def generate_heat_map(self, heatmap_data):
        plt.figure(figsize=(10, 8))

        models = self.get_paths_by_gt()
        x_labels = [models[i] for i in range(1, 99)]
        y_labels = [self.imagenet_classes[i] for i in range(1, 1000)]

        plt.figure(
            figsize=(20, 10)
        )  # You might need to adjust this depending on your screen and readability
        sns.heatmap(
            heatmap_data,
            xticklabels=x_labels,
            yticklabels=y_labels,
            cmap="YlGnBu",
            norm=LogNorm(vmin=heatmap_data.min().min(), vmax=heatmap_data.max().max()),
        )

        plt.xticks(rotation=45)  # Rotate x-axis labels to avoid overlap
        plt.yticks(rotation=0)  # Ensure y-axis labels are horizontal for readability

        plt.xlabel("Model Classes")
        plt.ylabel("Predicted Classes")

        plt.show()

    def mk_imagenet_map(self):
        LABELS_URL = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"

        response = requests.get(LABELS_URL)
        class_idx = response.json()
        self.imagenet_classes = {int(key): value[1] for key, value in class_idx.items()}


path = "resnet.csv"
# path = "res_small.csv"
heatmap_generator = DataFrameHeatMap(path)
heatmap_data = heatmap_generator.process_data()
heatmap_generator.generate_heat_map(heatmap_data)
