import os
import random
import time
import numpy as np
import pandas as pd
import torch
from scipy.special import expit, softmax
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
)
from sklearn.model_selection import train_test_split
from model import CNN, RNN, Autoencoder
from dataset import MitbihDataset, PtbdbDataset


def get_data(dataset_name, dataset_dir, seed):
    if dataset_name == "mitbih":
        # copied from baseline file from here
        df_train = pd.read_csv(f"{dataset_dir}/mitbih_train.csv", header=None)
        df_train = df_train.sample(frac=1)
        df_test = pd.read_csv(f"{dataset_dir}/mitbih_test.csv", header=None)

        Y = np.array(df_train[187].values).astype(np.int8)  # train + val
        X = np.array(df_train[list(range(187))].values)[..., np.newaxis]  # train + val

        Y_test = np.array(df_test[187].values).astype(np.int8)  # test
        X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]  # test
        # until here
    elif dataset_name == "ptbdb":
        # copied from baseline file from here
        df_1 = pd.read_csv(f"{dataset_dir}/ptbdb_normal.csv", header=None)
        df_2 = pd.read_csv(f"{dataset_dir}/ptbdb_abnormal.csv", header=None)
        df = pd.concat([df_1, df_2])

        df_train, df_test = train_test_split(
            df, test_size=0.2, random_state=seed, stratify=df[187]
        )

        Y = np.array(df_train[187].values).astype(np.int8)
        X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

        Y_test = np.array(df_test[187].values).astype(np.int8)
        X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]
        # until here
    else:
        raise Exception(f"Not a valid dataset_name {dataset_name}")

    return X, Y, X_test, Y_test


def get_data_loader(cfg, split, shuffle):
    dataset_name = cfg["dataset_name"]
    if dataset_name == "mitbih":
        Ds = MitbihDataset
    elif dataset_name == "ptbdb":
        Ds = PtbdbDataset
    else:
        raise Exception(f"Not a valid dataset_name {dataset_name}")

    dataset = Ds(dataset_dir=cfg["dataset_dir"], split=split, seed=cfg["seed"])
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=shuffle,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    return data_loader


def get_checkpoints_dir(cfg):
    model_name = cfg["model_name"]
    if "rnn" in model_name and cfg["rnn_bidirectional"]:
        model_name = "bidirectional_" + model_name
    checkpoints_dir = os.path.join(
        cfg["checkpoints_dir"],
        cfg["dataset_name"] + "_" + model_name + "_" + cfg["experiment_time"],
    )
    os.makedirs(checkpoints_dir, exist_ok=True)
    return checkpoints_dir


def write_and_print_new_log(new_log, cfg):
    print(new_log)

    checkpoints_dir = get_checkpoints_dir(cfg)
    log_path = os.path.join(checkpoints_dir, "logs.txt")
    with open(log_path, "a") as f:
        f.write(new_log + "\n")


def save_predictions_to_disk(all_y, all_yhat, split, cfg):
    checkpoints_dir = get_checkpoints_dir(cfg)
    predictions_path = os.path.join(checkpoints_dir, f"{split}_predictions.txt")
    if cfg["dataset_name"] == "mitbih":
        all_yhat_softmaxed = softmax(all_yhat, axis=1)
        df = pd.DataFrame(
            np.hstack((all_yhat_softmaxed, all_y.reshape(-1, 1))),
            columns=["prob_0", "prob_1", "prob_2", "prob_3", "prob_4", "label"],
        )
    else:
        logit_1 = all_yhat
        prob_1 = expit(logit_1)
        prob_0 = 1 - prob_1
        df = pd.DataFrame(
            np.hstack((prob_0, prob_1, all_y.reshape(-1, 1))),
            columns=["prob_0", "prob_1", "label"],
        )
    df.to_csv(predictions_path, index=False)


def get_model(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if "rnn" in cfg["model_name"]:
        model = RNN
    elif "cnn" in cfg["model_name"]:
        model = CNN
    elif "ae" in cfg["model_name"]:
        model = Autoencoder
    else:
        raise Exception(f"Not a valid model_name {cfg['model_name']}.")

    model = model(cfg).to(device)
    write_and_print_new_log(
        f"Total number of trainable parameters in {cfg['model_name']} model: "
        + str(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        cfg,
    )
    return model


def get_optimizer(cfg, model):
    return torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )


def get_scheduler(cfg, optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=cfg["lr_scheduler_patience"], verbose=True
    )


def set_seeds(cfg):
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed(cfg["seed"])


def save_checkpoint(model, cfg):
    write_and_print_new_log("Saving the best checkpoint...", cfg)
    checkpoints_dir = get_checkpoints_dir(cfg)
    checkpoint_dict = {"model_state_dict": model.state_dict()}
    torch.save(checkpoint_dict, os.path.join(checkpoints_dir, "best_checkpoint"))


def load_checkpoint(cfg):
    checkpoints_dir = get_checkpoints_dir(cfg)
    checkpoint_dict = torch.load(os.path.join(checkpoints_dir, "best_checkpoint"))
    model = get_model(cfg)
    model.load_state_dict(checkpoint_dict["model_state_dict"])
    return model


def evaluate_predictions(all_y, all_yhat, class_weights, cfg):
    sample_weights = np.array(
        [class_weights[int(label)] for label in all_y], dtype=np.float32
    )
    result_dict = {}
    if cfg["dataset_name"] == "mitbih":
        all_yhat_argmaxed = np.argmax(all_yhat, axis=1)
        result_dict["unbalanced_acc_score"] = accuracy_score(all_y, all_yhat_argmaxed)
        result_dict["balanced_acc_score"] = balanced_accuracy_score(
            all_y, all_yhat_argmaxed
        )
        result_dict["cross_entropy_loss"] = float(
            torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights))(
                torch.tensor(all_yhat), torch.tensor(all_y)
            )
        )
    elif cfg["dataset_name"] == "ptbdb":
        all_yhat_sigmoided = expit(all_yhat)
        all_yhat_argmaxed = 1 * (all_yhat_sigmoided >= 0.5)
        result_dict["unbalanced_acc_score"] = accuracy_score(all_y, all_yhat_argmaxed)
        result_dict["balanced_acc_score"] = balanced_accuracy_score(
            all_y, all_yhat_argmaxed
        )
        result_dict["roc_auc_score"] = roc_auc_score(all_y, all_yhat_sigmoided)
        result_dict["pr_auc_score"] = average_precision_score(all_y, all_yhat_sigmoided)
        result_dict["cross_entropy_loss"] = float(
            torch.nn.BCEWithLogitsLoss(weight=torch.tensor(sample_weights))(
                torch.tensor(all_yhat).squeeze(), torch.tensor(all_y).float()
            )
        )
    else:
        raise Exception(f"Not a valid dataset {cfg['dataset']}.")
    return result_dict


def pad_signals(signals, target_length):
    return torch.nn.functional.pad(signals, (0, 0, 0, target_length - signals.shape[1]))
