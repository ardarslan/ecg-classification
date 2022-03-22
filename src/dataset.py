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
    
    def prepare_X_y(self, X, Y, X_test, Y_test, val_ratio):
        X_train_val = X
        y_train_val = Y
        X_test = X_test
        y_test = Y_test
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_ratio, random_state=seed, stratify=y_train_val)

        if split == "train":
            self.X = X_train
            self.y = y_train
        elif split == "train_val":
            self.X = X_train_val
            self.y = y_train_val
        elif split == "val":
            self.X = X_val
            self.y = y_val
        elif split == "test":
            self.X = X_test
            self.y = y_test
        else:
            raise Exception(f"Not a valid split {split}.")
    
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

        # copied from baseline file from here
        df_train = pd.read_csv(f"{dataset_dir}/mitbih_train.csv", header=None)
        df_train = df_train.sample(frac=1)
        df_test = pd.read_csv(f"{dataset_dir}/mitbih_test.csv", header=None)

        Y = np.array(df_train[187].values).astype(np.int8)  # train + val
        X = np.array(df_train[list(range(187))].values)[..., np.newaxis]  # train + val

        Y_test = np.array(df_test[187].values).astype(np.int8)  # test
        X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]  # test
        # until here

        self.prepare_X_y(X, Y, X_test, Y_test, val_ratio=0.15)


class PtbdbDataset(Dataset):
    def __init__(self, dataset_dir, split, seed):
        """ Initialization of the Ptbdb dataset.

        Args:
            dataset_dir (str): path to the directory in which the .csv files are located
            split (str): train or val
        """
        super().__init__(dataset_dir=dataset_dir, split=split, seed=seed)
        
        # copied from baseline file from here
        df_1 = pd.read_csv(f"{dataset_dir}/ptbdb_normal.csv", header=None)
        df_2 = pd.read_csv(f"{dataset_dir}/ptbdb_abnormal.csv", header=None)
        df = pd.concat([df_1, df_2])

        df_train, df_test = train_test_split(df, test_size=0.2, random_state=seed, stratify=df[187])

        Y = np.array(df_train[187].values).astype(np.int8)
        X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

        Y_test = np.array(df_test[187].values).astype(np.int8)
        X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]
        # until here

        self.prepare_X_y(X, Y, X_test, Y_test, val_ratio=0.20)