import os
import random
import time
import numpy as np
import torch
import yaml
from scipy.special import expit
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from model import CNN, RNN
from dataset import MitbihDataset, PtbdbDataset


def read_config(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["experiment_time"] = str(int(time.time()))
    return cfg


def get_data_loader(cfg, split):
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
        shuffle=(split != "test"),
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=True
    )
    return data_loader


def get_model(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name_model_mapping = {
        "rnn": RNN,
        "cnn": CNN
    }
    model = model_name_model_mapping[cfg["model_name"]](cfg).to(device)
    print(
        f"Total number of parameters in {cfg['model_name']} model: "
        + str(sum(p.numel() for p in model.parameters() if p.requires_grad))
    )
    return model


def get_optimizer(cfg, model):
    return torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])


def get_scheduler(cfg, optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)


def set_seeds(cfg):
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed(cfg["seed"])


def save_checkpoint(model, optimizer, train_loss_at_early_stop, epoch, cfg):
    print("Saving the best checkpoint...")
    checkpoints_dir = os.path.join(cfg["checkpoints_dir"], cfg["experiment_time"])
    os.makedirs(checkpoints_dir, exist_ok=True)
    checkpoint_dict = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss_at_early_stop": train_loss_at_early_stop,
        "epoch": epoch
    }
    torch.save(checkpoint_dict, os.path.join(checkpoints_dir, "best_checkpoint"))


def load_checkpoint(cfg):
    checkpoints_dir = os.path.join(cfg["checkpoints_dir"], cfg["experiment_time"])
    checkpoint_dict = torch.load(os.path.join(checkpoints_dir, "best_checkpoint"))
    model = get_model(cfg)
    model.load_state_dict(checkpoint_dict["model_state_dict"])
    optimizer = get_optimizer(cfg, model)
    optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
    train_loss_at_early_stop = checkpoint_dict["train_loss_at_early_stop"]
    epoch = checkpoint_dict["epoch"]
    return model, optimizer, train_loss_at_early_stop, epoch


def evaluate_predictions(all_y, all_yhat, class_weights, cfg):
    sample_weights = np.array([class_weights[int(label)] for label in all_y], dtype=np.float32)
    result_dict = {}
    if cfg["dataset_name"] == "mitbih":
        result_dict["unbalanced_acc_score"] = accuracy_score(all_y, np.argmax(all_yhat, axis=1))
        result_dict["balanced_acc_score"] = accuracy_score(all_y, np.argmax(all_yhat, axis=1), sample_weight=sample_weights)
        result_dict["cross_entropy_loss"] = float(torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights))(torch.tensor(all_yhat), torch.tensor(all_y)))
    elif cfg["dataset_name"] == "ptbdb":
        all_yhat_sigmoided = expit(all_yhat)
        result_dict["roc_auc_score"] = roc_auc_score(all_y, all_yhat_sigmoided)
        result_dict["pr_auc_score"] = average_precision_score(all_y, all_yhat_sigmoided)
        result_dict["cross_entropy_loss"] = float(torch.nn.BCEWithLogitsLoss(weight=torch.tensor(sample_weights).unsqueeze(-1))(torch.tensor(all_yhat), torch.tensor(all_y).float()))
    else:
        raise Exception(f"Not a valid dataset {cfg['dataset']}.")
    return result_dict
