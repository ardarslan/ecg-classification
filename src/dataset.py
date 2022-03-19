import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, split, seed):
        """ Initialization of the abstract class Dataset.
        
        Args:
            dataset_dir (str): path to the directory in which the .csv files are located
            split (str): train or val
        """
        super().__init__()
    
    def __len__(self):
        """ Returns the length of the dataset. """
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        """ Returns an item of the dataset.

        Args:
            idx (int): data ID.
        """
        return {
                "X": self.X[idx, :],
                "y": self.y[idx]
               }


class MitbihDataset(Dataset):
    def __init__(self, dataset_dir, split, seed):
        """ Initialization of the Mitbih dataset.

        Args:
            dataset_dir (str): path to the directory in which the .csv files are located
            split (str): train or val
        """
        super().__init__(dataset_dir=dataset_dir, split=split, seed=seed)
        if split in ["train", "val", "train_val"]:
            df_train_val = pd.read_csv(f"{dataset_dir}/mitbih_train.csv", header=None)
            df_train_val = df_train_val.sample(frac=1)
            df_train, df_val = train_test_split(df_train_val, test_size=0.25, random_state=seed, stratify=df_train_val[187])
            if split == "train":
                self.y = np.array(df_train[187].values).astype(np.int8)
                self.X = np.array(df_train[list(range(187))].values)[..., np.newaxis]
                self.class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(self.y.ravel()), y=self.y.ravel()).astype(np.float32)
            elif split == "val":
                self.y = np.array(df_val[187].values).astype(np.int8)
                self.X = np.array(df_val[list(range(187))].values)[..., np.newaxis]
            elif split == "train_val":
                self.y = np.array(df_train_val[187].values).astype(np.int8)
                self.X = np.array(df_train_val[list(range(187))].values)[..., np.newaxis]
                self.class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(self.y.ravel()), y=self.y.ravel()).astype(np.float32)
        elif split == "test":
            df_test = pd.read_csv(f"{dataset_dir}/mitbih_test.csv", header=None)
            self.y = np.array(df_test[187].values).astype(np.int8)
            self.X = np.array(df_test[list(range(187))].values)[..., np.newaxis]


class PtbdbDataset(Dataset):
    def __init__(self, dataset_dir, split, seed):
        """ Initialization of the Ptbdb dataset.

        Args:
            dataset_dir (str): path to the directory in which the .csv files are located
            split (str): train or val
        """
        super().__init__(dataset_dir=dataset_dir, split=split, seed=seed)
        df_1 = pd.read_csv(f"{dataset_dir}/ptbdb_normal.csv", header=None)
        df_2 = pd.read_csv(f"{dataset_dir}/ptbdb_abnormal.csv", header=None)
        df = pd.concat([df_1, df_2])

        df_train, df_test = train_test_split(df, test_size=0.2, random_state=seed, stratify=df[187])
        df_train, df_val = train_test_split(df_train, test_size=0.25, random_state=seed, stratify=df_train[187])
        
        if split == "train":
            self.y = np.array(df_train[187].values).astype(np.int8).reshape(-1, 1)
            self.X = np.array(df_train[list(range(187))].values)[..., np.newaxis]
            self.class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(self.y.ravel()), y=self.y.ravel()).astype(np.float32)
        elif split == "val":
            self.y = np.array(df_val[187].values).astype(np.int8).reshape(-1, 1)
            self.X = np.array(df_val[list(range(187))].values)[..., np.newaxis]
        elif split == "train_val":
            self.y = np.vstack((np.array(df_train[187].values).astype(np.int8).reshape(-1, 1),
                                np.array(df_val[187].values).astype(np.int8).reshape(-1, 1)))
            self.X = np.vstack((np.array(df_train[list(range(187))].values)[..., np.newaxis],
                                np.array(df_val[list(range(187))].values)[..., np.newaxis]))
            self.class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(self.y.ravel()), y=self.y.ravel()).astype(np.float32)
        elif split == "test":
            self.y = np.array(df_test[187].values).astype(np.int8).reshape(-1, 1)
            self.X = np.array(df_test[list(range(187))].values)[..., np.newaxis]