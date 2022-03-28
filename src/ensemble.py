import os
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
)

from typing import List, Tuple

from utils import (
    set_seeds,
    get_argument_parser,
    write_and_print_new_log,
    save_predictions_to_disk,
    get_checkpoints_dir,
)


def get_prediction_folders(checkpoints_dir: str) -> List[str]:
    files = os.listdir(checkpoints_dir)
    # Avoid ensembling over ensembles.
    files = [file for file in files if "ensemble" not in file]
    return [
        file for file in files if os.path.isdir(os.path.join(checkpoints_dir, file))
    ]


def get_model_name_from_folder(folder: str) -> str:
    # Remove dataset identifiers.
    model_name = folder.replace("mitbih_", "").replace("ptbdb_", "")
    # Remove timestamp at end.
    return model_name[:-11]


def predict_max_prob(preds: pd.DataFrame) -> pd.DataFrame:
    max_prob_preds = preds.copy()
    max_prob_preds["pred"] = max_prob_preds[preds.columns.difference(["label"])].idxmax(
        axis=1
    )
    max_prob_preds = max_prob_preds[["pred", "label"]]
    max_prob_preds["pred"] = max_prob_preds["pred"].apply(lambda s: float(s[-1]))
    return max_prob_preds


def load_predictions(
    dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_pred = pd.read_csv(os.path.join(dir, "train_predictions.txt"))
    val_pred = pd.read_csv(os.path.join(dir, "val_predictions.txt"))
    test_pred = pd.read_csv(os.path.join(dir, "test_predictions.txt"))

    train_pred = predict_max_prob(train_pred)
    val_pred = predict_max_prob(val_pred)
    test_pred = predict_max_prob(test_pred)

    return train_pred, val_pred, test_pred


def create_ensemble_datasets(
    cfg: dict, excluded_models: list[dir]
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    checkpoints_dir = cfg["checkpoints_dir"]
    dataset_name = cfg["dataset_name"]

    preds_folders = get_prediction_folders(checkpoints_dir)
    filtered_preds_folders = [
        folder for folder in preds_folders if dataset_name in folder
    ]
    model_preds_dirs = [
        (get_model_name_from_folder(folder), os.path.join(checkpoints_dir, folder))
        for folder in filtered_preds_folders
    ]

    ensembled_models = []
    train_preds, val_preds, test_preds = None, None, None
    for model_name, dir in model_preds_dirs:
        if model_name in excluded_models:
            continue

        if train_preds is None:
            train_preds, val_preds, test_preds = load_predictions(dir)
            train_preds.rename(columns={"pred": f"{model_name}_pred"}, inplace=True)
            val_preds.rename(columns={"pred": f"{model_name}_pred"}, inplace=True)
            test_preds.rename(columns={"pred": f"{model_name}_pred"}, inplace=True)
            continue

        train_preds_new, val_preds_new, test_preds_new = load_predictions(dir)
        train_preds[f"{model_name}_pred"] = train_preds_new["pred"]
        val_preds[f"{model_name}_pred"] = val_preds_new["pred"]
        test_preds[f"{model_name}_pred"] = test_preds_new["pred"]

        ensembled_models.append(model_name)

    X_train = train_preds[train_preds.columns.difference(["label"])]
    y_train = train_preds["label"]
    X_val = val_preds[val_preds.columns.difference(["label"])]
    y_val = val_preds["label"]
    X_test = test_preds[test_preds.columns.difference(["label"])]
    y_test = test_preds["label"]

    write_and_print_new_log("Ensembling: " + ", ".join(ensembled_models), cfg)

    return X_train, y_train, X_val, y_val, X_test, y_test


def get_majority_prediction(X: pd.DataFrame) -> pd.Series:
    return X.mode(axis=1)[0]


def save_majority_preds_to_disk(labels: np.ndarray, preds: np.ndarray, split, cfg):
    checkpoints_dir = get_checkpoints_dir(cfg)
    predictions_path = os.path.join(checkpoints_dir, f"{split}_predictions.txt")
    df = pd.DataFrame(
        np.hstack((preds, labels.reshape(-1, 1))),
        columns=["preds", "label"],
    )
    df.to_csv(predictions_path, index=False)


def eval_ensemble_preds(labels: np.ndarray, preds: np.ndarray, cfg: dict) -> dict:
    result_dict = {}

    # Add extra metrics for binary class. problem.
    if cfg["dataset_name"] == "ptbdb":
        result_dict["roc_auc_score"] = roc_auc_score(labels, preds[:, 1])
        result_dict["pr_auc_score"] = average_precision_score(labels, preds[:, 1])

    # If we have more than 1 dimension, predictions are given as probabilities
    # and so map them to class predictions.
    if preds.shape[1] > 1:
        preds = np.argmax(preds, axis=1)
    result_dict["unbalanced_acc_score"] = accuracy_score(labels, preds)
    result_dict["balanced_acc_score"] = balanced_accuracy_score(labels, preds)

    return result_dict


def run_ensemble(cfg: dict) -> None:

    X_train, y_train, X_val, y_val, X_test, y_test = create_ensemble_datasets(
        cfg, excluded_models=["baseline_cnn"]
    )

    if "majority_ensemble" == cfg["model_name"]:
        y_hat_train = get_majority_prediction(X_train).to_numpy().reshape(-1, 1)
        y_hat_val = get_majority_prediction(X_val).to_numpy().reshape(-1, 1)
        y_hat_test = get_majority_prediction(X_test).to_numpy().reshape(-1, 1)

        save_majority_preds_to_disk(y_train.to_numpy(), y_hat_train, "train", cfg)
        save_majority_preds_to_disk(y_val.to_numpy(), y_hat_val, "val", cfg)
        save_majority_preds_to_disk(y_test.to_numpy(), y_hat_test, "test", cfg)

        # Need to map predictions to probabilities for AUROC and AUPRC. We
        # simply predict probability 1.0 for the voted majority class.
        if "ptbdb" == cfg["dataset_name"]:
            y_hat_test = np.hstack((1 - y_hat_test, y_hat_test))
    elif "log_reg_ensemble" == cfg["model_name"]:
        log_reg = LogisticRegression(random_state=cfg["seed"], max_iter=1e5)
        log_reg.fit(X_train, y_train)

        y_hat_train = log_reg.predict_proba(X_train)
        y_hat_val = log_reg.predict_proba(X_val)
        y_hat_test = log_reg.predict_proba(X_test)

        if cfg["dataset_name"] == "ptbdb":
            y_hat_train_save = y_hat_train[:, 1:]
            y_hat_val_save = y_hat_val[:, 1:]
            y_hat_test_save = y_hat_test[:, 1:]
        else:
            y_hat_train_save = y_hat_train
            y_hat_val_save = y_hat_val
            y_hat_test_save = y_hat_test

        save_predictions_to_disk(
            y_train.to_numpy(), y_hat_train_save, "train", cfg, use_logits=False
        )
        save_predictions_to_disk(
            y_val.to_numpy(), y_hat_val_save, "val", cfg, use_logits=False
        )
        save_predictions_to_disk(
            y_test.to_numpy(), y_hat_test_save, "test", cfg, use_logits=False
        )

    test_loss_dict = eval_ensemble_preds(y_test.to_numpy(), y_hat_test, cfg)

    new_log = f"Test {cfg['dataset_name']} | " + ", ".join(
        [
            f"{loss_function}: {np.round(loss_value, 3)}"
            for loss_function, loss_value in test_loss_dict.items()
        ]
    )
    write_and_print_new_log(new_log, cfg)


if __name__ == "__main__":
    cfg = get_argument_parser().parse_args().__dict__
    cfg["experiment_time"] = str(int(time.time()))

    set_seeds(cfg)

    write_and_print_new_log(
        f"Dataset name: {cfg['dataset_name']}, "
        + f"Model name: {cfg['model_name']}, "
        + f"Transfer learning: {cfg['transfer_learning']}, "
        + f"RNN freeze: {cfg['rnn_freeze']}, "
        + f"RNN Bidirectional: {cfg['rnn_bidirectional']}, "
        + f"RNN Num Layers: {cfg['rnn_num_layers']}",
        cfg,
    )

    run_ensemble(cfg)
