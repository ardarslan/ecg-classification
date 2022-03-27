import os
import random
import argparse
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
from model import CNN, RNN, Autoencoder, InceptionNet, SharedMLPOverRNN
from dataset import MitbihDataset, PtbdbDataset


def get_data_loader(cfg, split, shuffle):
    dataset_name = cfg["dataset_name"]
    if dataset_name == "mitbih":
        Ds = MitbihDataset
    elif dataset_name == "ptbdb":
        Ds = PtbdbDataset
    else:
        raise Exception(f"Not a valid dataset_name {dataset_name}")

    dataset = Ds(dataset_dir=cfg["dataset_dir"], split=split, seed=cfg["seed"], cfg=cfg)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=shuffle,
        num_workers=cfg["num_workers"],
        pin_memory=False,
        drop_last=False,
    )
    return data_loader


def get_checkpoints_dir(cfg):
    model_name = cfg["model_name"]
    if "rnn" in model_name and cfg["rnn_bidirectional"]:
        model_name = "bidirectional_" + model_name
    if cfg["transfer_learning"]:
        dataset_name = "ptbdb"
    else:
        dataset_name = cfg["dataset_name"]
    checkpoints_dir = os.path.join(
        cfg["checkpoints_dir"],
        dataset_name + "_" + model_name + "_" + cfg["experiment_time"],
    )
    os.makedirs(checkpoints_dir, exist_ok=True)
    return checkpoints_dir


def write_and_print_new_log(new_log, cfg):
    print(new_log)

    checkpoints_dir = get_checkpoints_dir(cfg)
    log_path = os.path.join(checkpoints_dir, "logs.txt")
    with open(log_path, "a") as f:
        f.write(new_log + "\n")


def save_predictions_to_disk(all_y, all_yhat, split, cfg, use_logits):
    checkpoints_dir = get_checkpoints_dir(cfg)
    predictions_path = os.path.join(checkpoints_dir, f"{split}_predictions.txt")
    if cfg["dataset_name"] == "mitbih":
        if use_logits:
            all_yhat_probs = softmax(all_yhat, axis=1)
        else:
            all_yhat_probs = all_yhat
        columns = ["prob_0", "prob_1", "prob_2", "prob_3", "prob_4", "label"]
    else:
        if use_logits:
            logit_1 = all_yhat
            prob_1 = expit(logit_1)
        else:
            prob_1 = all_yhat

        prob_0 = 1 - prob_1
        all_yhat_probs = np.hstack((prob_0, prob_1))
        columns = ["prob_0", "prob_1", "label"]
    df = pd.DataFrame(
        np.hstack((all_yhat_probs, all_y.reshape(-1, 1))), columns=columns
    )
    df.to_csv(predictions_path, index=False)


def get_model(cfg):
    if "sharedmlpover" in cfg["model_name"]:
        model = SharedMLPOverRNN
    elif "inception" in cfg["model_name"]:
        model = InceptionNet
    elif "rnn" in cfg["model_name"]:
        model = RNN
    elif "cnn" in cfg["model_name"]:
        model = CNN
    elif "ae" in cfg["model_name"]:
        model = Autoencoder
    else:
        raise Exception(f"Not a valid model_name {cfg['model_name']}.")

    model = model(cfg).to(cfg["device"])
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


def evaluate_predictions(all_y, all_yhat, class_weights, cfg, use_logits):
    sample_weights = torch.tensor(
        [class_weights[int(label)] for label in all_y],
        dtype=torch.float,
        device=cfg["device"],
    )
    result_dict = {}
    if cfg["dataset_name"] == "mitbih":
        if use_logits:
            all_yhat_probs = softmax(all_yhat, axis=1)
            result_dict["cross_entropy_loss"] = float(
                torch.nn.CrossEntropyLoss(weight=class_weights)(
                    torch.tensor(all_yhat, device=cfg["device"], dtype=torch.float),
                    torch.tensor(all_y, device=cfg["device"], dtype=torch.long),
                )
            )
        else:
            all_yhat_probs = all_yhat
        all_yhat_argmaxed = np.argmax(all_yhat_probs, axis=1)
        result_dict["unbalanced_acc_score"] = accuracy_score(all_y, all_yhat_argmaxed)
        result_dict["balanced_acc_score"] = balanced_accuracy_score(
            all_y, all_yhat_argmaxed
        )
    elif cfg["dataset_name"] == "ptbdb":
        if use_logits:
            all_yhat_probs = expit(all_yhat)
            result_dict["cross_entropy_loss"] = float(
                torch.nn.BCEWithLogitsLoss(weight=sample_weights)(
                    torch.tensor(
                        all_yhat, device=cfg["device"], dtype=torch.float
                    ).squeeze(),
                    torch.tensor(all_y, device=cfg["device"], dtype=torch.float),
                )
            )
        else:
            all_yhat_probs = all_yhat
        all_yhat_argmaxed = 1 * (all_yhat_probs >= 0.5)
        result_dict["unbalanced_acc_score"] = accuracy_score(all_y, all_yhat_argmaxed)
        result_dict["balanced_acc_score"] = balanced_accuracy_score(
            all_y, all_yhat_argmaxed
        )
        result_dict["roc_auc_score"] = roc_auc_score(all_y, all_yhat_probs)
        result_dict["pr_auc_score"] = average_precision_score(all_y, all_yhat_probs)
    else:
        raise Exception(f"Not a valid dataset {cfg['dataset']}.")
    return result_dict


def pad_signals(signals, target_length):
    return torch.nn.functional.pad(signals, (0, 0, 0, target_length - signals.shape[1]))


def get_argument_parser():
    parser = argparse.ArgumentParser(description="Arguments for running the script")

    parser.add_argument("--dataset_dir", type=str, default="../data")
    parser.add_argument("--checkpoints_dir", type=str, default="../checkpoints")
    parser.add_argument("--dataset_name", type=str, default="mitbih")  # mitbih, ptbdb
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--num_workers", type=int, default=1
    )  # 0 means use the same thread for data processing
    parser.add_argument(
        "--model_name", type=str, default="vanilla_rnn"
    )  # vanilla_rnn, lstm_rnn, gru_rnn, vanilla_cnn, residual_cnn
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--lr", type=int, default=0.001)
    parser.add_argument("--weight_decay", type=int, default=0.0)
    parser.add_argument("--use_lr_scheduler", type=int, default=True)
    parser.add_argument("--lr_scheduler_patience", type=int, default=5)
    parser.add_argument("--early_stop_patience", type=int, default=15)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--gradient_max_norm", type=int, default=5.0)
    parser.add_argument("--transfer_learning", action="store_true")

    # rnn configs
    parser.add_argument("--rnn_hidden_size", type=int, default=128)
    parser.add_argument("--rnn_num_layers", type=int, default=1)
    parser.add_argument("--rnn_bidirectional", action="store_true")
    parser.add_argument("--rnn_dropout", type=float, default=0.0)
    parser.add_argument(
        "--rnn_freeze",
        type=str,
        default="never",
        help=""" - permanent: train only a new FCNN on top of RNN, """
        """ - temporary: train only a new FCNN on top of RNN """
        """ for 'rnn_freeze_num_epochs', after that start training the """
        """ RNN as well, """
        """ - never: both RNN and FCNN will be trained from the """
        """ the beginning of finetuning.""",
    )
    parser.add_argument("--rnn_freeze_num_epochs", type=int, default=20)

    # cnn configs
    parser.add_argument("--cnn_num_layers", type=int, default=4)
    parser.add_argument("--cnn_num_channels", type=int, default=64)

    # autoencoder configs
    parser.add_argument("--ae_latent_dim", type=int, default=30)
    parser.add_argument("--ae_output_dir", type=str, default="data-encoded")

    return parser
