import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from utils import get_data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, split, seed):
        """Initialization of the abstract class Dataset.

        Args:
            dataset_dir (str): path to the directory in which the .csv files are located
            split (str): train or val
        """
        super().__init__()
        self.dataset_dir = dataset_dir
        self.split = split
        self.seed = seed

    def prepare_X_y(self, X, Y, X_test, Y_test, val_ratio):
        X_train_val = X
        y_train_val = Y
        X_test = X_test
        y_test = Y_test
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=val_ratio,
            random_state=self.seed,
            stratify=y_train_val,
        )

        if self.split == "train":
            self.X = X_train
            self.y = y_train
            self.class_weights = compute_class_weight(
                class_weight="balanced",
                classes=np.unique(self.y.ravel()),
                y=self.y.ravel(),
            ).astype(np.float32)
        elif self.split == "train_val":
            self.X = X_train_val
            self.y = y_train_val
            self.class_weights = compute_class_weight(
                class_weight="balanced",
                classes=np.unique(self.y.ravel()),
                y=self.y.ravel(),
            ).astype(np.float32)
        elif self.split == "val":
            self.X = X_val
            self.y = y_val
        elif self.split == "test":
            self.X = X_test
            self.y = y_test
        else:
            raise Exception(f"Not a valid split {split}.")

    def __len__(self):
        """Returns the length of the dataset."""
        return self.X.shape[0]

    def __getitem__(self, idx):
        """Returns an item of the dataset.

        Args:
            idx (int): data ID.
        """
        return {"X": self.X[idx, :], "y": self.y[idx]}


class MitbihDataset(Dataset):
    def __init__(self, dataset_dir, split, seed):
        """Initialization of the Mitbih dataset.

        Args:
            dataset_dir (str): path to the directory in which the .csv files are located
            split (str): train or val
        """
        super().__init__(dataset_dir=dataset_dir, split=split, seed=seed)

        X, Y, X_test, Y_test = get_data(
            dataset_name="mitbih", dataset_dir=dataset_dir, seed=seed
        )

        self.prepare_X_y(X, Y, X_test, Y_test, val_ratio=0.15)


class PtbdbDataset(Dataset):
    def __init__(self, dataset_dir, split, seed):
        """Initialization of the Ptbdb dataset.

        Args:
            dataset_dir (str): path to the directory in which the .csv files are located
            split (str): train or val
        """
        super().__init__(dataset_dir=dataset_dir, split=split, seed=seed)

        X, Y, X_test, Y_test = get_data(
            dataset_name="mitbih", dataset_dir=dataset_dir, seed=seed
        )

        self.prepare_X_y(X, Y, X_test, Y_test, val_ratio=0.20)
