import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from matplotlib.colors import LogNorm

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


class CSVAnalyzer:
    def __init__(self, file_path):
        self.categories = {
            # "Non-Biological": {
            "helmet": [518, 560, 570],
            "mailbox": [637],
            "bathtub": [435, 876],
            "mug": [504],
            "phone": [487, 528, 707],
            # "Biological": {
            "fish": [1, 5, 147, 390, 391, 392, 393, 394, 395, 396, 397],
            "elephant": [385, 386],
            "butterfly": [322, 323, 324, 326],
            "bird": [12, 13, 14, 15, 16, 17, 18, 19, 20],
            "bear": [105, 294, 295, 296, 297, 387, 388],
        }

        self.df = pd.read_csv(file_path)
        self.mk_imagenet_map()

        self.df["viewpoint"] = self.df["path"].apply(
            lambda x: x.split("/")[-1].split("_")[-1].split(".")[0]
        )
        self.df["shape"] = (
            self.df["path"]
            .apply(lambda x: x.split("/")[-1].split("_")[0])
            .replace("\d+", "", regex=True)
        )
        self.df["texture"] = (
            self.df["path"]
            .apply(lambda x: x.split("/")[-1].split("_")[1])
            .replace("\d+", "", regex=True)
        )

        self.df = self.df[self.df["texture"] != "black"]

        for category, cols in self.categories.items():
            cols = [f"out{i}" for i in cols]
            self.df[category.lower() + "_max"] = self.df[cols].max(axis=1)

        self.df = self.df[[c for c in self.df.columns if not ("out" in c)]]

        self.df["shape_prob"] = self.df.apply(lambda x: x[f"{x['shape']}_max"], axis=1)
        self.df["texture_prob"] = self.df.apply(
            lambda x: x[f"{x['texture']}_max"], axis=1
        )

        self.df["relevant_categories"] = self.df[
            [c for c in self.df.columns if "_max" in c]
        ].sum(axis=1)

        self.df["other_prob"] = (
            self.df["relevant_categories"]
            - self.df["shape_prob"]
            + self.df["texture_prob"]
        )
        self.df["vote"] = self.df.apply(
            lambda x: "other"
            if x["other_prob"] > x["shape_prob"] and x["other_prob"] > x["texture_prob"]
            else ("shape" if x["shape_prob"] > x["texture_prob"] else "texture"),
            axis=1,
        )

    def plt_stacked_bar(self):
        fig, axs = plt.subplots(2, 1)

        for ax, X in zip(axs, ["shape", "texture"]):
            votes = {
                k: {x: len(ssf["vote"]) for x, ssf in sf.groupby(X)}
                for k, sf in self.df.groupby("vote")
            }

            if "texture" not in votes:
                votes["texture"] = {k: 0 for k in votes["shape"].keys()}

            print(votes)

            def bar(label, bottom=None):
                x, y = votes[label].keys(), votes[label].values()
                ax.bar(list(x), list(y), label=label, bottom=list(bottom))

            bar(label="other")
            bar(bottom=votes["other"].values(), label="shape")
            bar(bottom=votes["shape"].values(), label="texture")

            ax.legend(loc="lower left")
            ax.set(
                title=f"Vote Distribution by {X.upper()} Ground Truth",
                xlabel=f"{X.upper()} Ground Truth",
                ylabel="(n) images",
            )

        plt.tight_layout()
        plt.show()

    def plt_votes(self):
        votes = {
            k: sum([v == k for v in self.df["vote"]])
            for k in ["other", "shape", "texture"]
        }
        plt.bar(votes.keys(), votes.values())
        plt.title(f"Vote Distribution | {votes}")
        plt.show()

    def plt_violin(self):
        shape_probs, texture_probs = (
            self.df["shape_prob"].tolist(),
            self.df["texture_prob"].tolist(),
        )

        # sns.boxplot( x=["shape"] * len(shape_probs) + ["texture"] * len(texture_probs), y=shape_probs + texture_probs,)
        sns.violinplot(
            x=["shape"] * len(shape_probs) + ["texture"] * len(texture_probs),
            y=shape_probs + texture_probs,
            inner="box",
            bw=0.00001,
        )
        plt.title("Shape vs Texture Probabilities")
        plt.show()

        quit()

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
A = CSVAnalyzer(path)
A.plt_stacked_bar()
