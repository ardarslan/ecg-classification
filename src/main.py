import os
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from dataset import get_data
from utils import (
    get_argument_parser,
    get_checkpoints_dir,
    set_seeds,
    get_model,
    get_optimizer,
    get_scheduler,
    get_data_loader,
    save_checkpoint,
    load_checkpoint,
    evaluate_predictions,
    write_and_print_new_log,
    save_predictions_to_disk,
    pad_signals,
)


def train_epoch(model, optimizer, train_data_loader, class_weights, cfg):
    model.train()
    all_y = []
    all_yhat = []
    for batch in train_data_loader:
        optimizer.zero_grad()
        X, y = (
            batch["X"].float().to(cfg["device"]),
            batch["y"].to(cfg["device"]),
        )
        yhat = model(X)
        if cfg["dataset_name"] == "mitbih":
            cross_entropy_loss = torch.nn.CrossEntropyLoss(weight=class_weights)(
                yhat, y.long()
            )
        elif cfg["dataset_name"] == "ptbdb":
            sample_weights = torch.tensor(
                [class_weights[int(label)] for label in y],
                dtype=torch.float,
                device=cfg["device"],
            )
            cross_entropy_loss = torch.nn.BCEWithLogitsLoss(weight=sample_weights)(
                yhat.squeeze(), y.float()
            )
        else:
            raise Exception(f"Not a valid dataset {cfg['dataset_name']}.")
        all_y.append(y.detach().cpu().numpy())
        all_yhat.append(yhat.detach().cpu().numpy())
        cross_entropy_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_max_norm"])
        optimizer.step()
    all_y = np.concatenate(all_y, axis=0)
    all_yhat = np.concatenate(all_yhat, axis=0).astype(np.float32)
    train_loss_dict = evaluate_predictions(
        all_y, all_yhat, class_weights, cfg, use_logits=True
    )
    return train_loss_dict


def train_ae_epoch(model, optimizer, train_data_loader, cfg):
    """Train epoch for Autoencoder."""
    model.train()
    total_mse_loss = 0.0
    for batch in train_data_loader:
        optimizer.zero_grad()

        X = batch["X"].float().to(cfg["device"])
        # Pad to large enough multiple of 2 so that no loss in dimensionality
        # through Autoencoder.
        X = pad_signals(X, 192)
        Xhat = model(X)

        mse_loss = torch.nn.MSELoss()(Xhat, X.permute(0, 2, 1))
        total_mse_loss += float(mse_loss)
        mse_loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_max_norm"])
        optimizer.step()
    return {"mse_loss": total_mse_loss / len(train_data_loader)}


def evaluation_epoch(
    model, evaluation_data_loader, class_weights, split, cfg, save_to_disk=False
):
    model.eval()
    with torch.no_grad():
        all_y = []
        all_yhat = []
        for batch in evaluation_data_loader:
            X, y = (
                batch["X"].float().to(cfg["device"]),
                batch["y"].to(cfg["device"]),
            )
            yhat = model(X)
            all_y.append(y.detach().cpu().numpy())
            all_yhat.append(yhat.detach().cpu().numpy())
        all_y = np.concatenate(all_y, axis=0)
        all_yhat = np.concatenate(all_yhat, axis=0).astype(np.float32)
        if save_to_disk:
            save_predictions_to_disk(all_y, all_yhat, split, cfg, use_logits=True)
        eval_loss_dict = evaluate_predictions(
            all_y, all_yhat, class_weights, cfg, use_logits=True
        )
    return eval_loss_dict


def evaluation_ae_epoch(model, evaluation_data_loader):
    model.eval()
    total_mse_loss = 0.0
    with torch.no_grad():
        for batch in evaluation_data_loader:
            X = batch["X"].float().to(cfg["device"])
            # Pad to large enough multiple of 2 so that no loss in dimensionality
            # through Autoencoder.
            X = pad_signals(X, 192)
            Xhat = model(X)

            mse_loss = torch.nn.MSELoss()(Xhat, X.permute(0, 2, 1))
            total_mse_loss += float(mse_loss.float())
    return {"mse_loss": total_mse_loss / len(evaluation_data_loader)}


def should_unfreeze_params(cfg, epoch):
    return (
        cfg["transfer_learning"]
        and cfg["dataset_name"] == "ptbdb"
        and cfg["rnn_freeze"] == "temporary"
        and cfg["rnn_freeze_num_epochs"] == epoch
    )


def train(cfg, model, train_split, validation_split):
    if not model:
        model = get_model(cfg)
    optimizer = get_optimizer(cfg, model)

    if cfg["use_lr_scheduler"]:
        scheduler = get_scheduler(cfg, optimizer)

    train_data_loader = get_data_loader(cfg, split=train_split, shuffle=True)
    class_weights = train_data_loader.dataset.class_weights
    val_data_loader = get_data_loader(cfg, split=validation_split, shuffle=False)

    best_val_loss = np.inf
    early_stop_counter = 0
    for epoch in range(cfg["max_epochs"]):
        # if we are in the second training part of transfer learning task and
        # the rule is to unfreeze RNN after rnn_freeze_num_epochs:
        if should_unfreeze_params(cfg, epoch):
            write_and_print_new_log("Unfreezing weights...", cfg)
            for param in model.parameters():
                param.requires_grad = True

        # train
        if "ae" in cfg["model_name"]:
            train_loss_dict = train_ae_epoch(model, optimizer, train_data_loader, cfg)
        else:
            train_loss_dict = train_epoch(
                model, optimizer, train_data_loader, class_weights, cfg
            )
        new_log = f"Train {cfg['dataset_name']} | Epoch: {epoch+1}, " + ", ".join(
            [
                f"{loss_function}: {np.round(loss_value, 3)}"
                for loss_function, loss_value in train_loss_dict.items()
            ]
        )
        write_and_print_new_log(new_log, cfg)

        # validate
        if "ae" in cfg["model_name"]:
            val_loss_dict = evaluation_ae_epoch(model, val_data_loader)
            current_val_loss = val_loss_dict["mse_loss"]
        else:
            val_loss_dict = evaluation_epoch(
                model, val_data_loader, class_weights, "val", cfg, save_to_disk=False
            )
            current_val_loss = val_loss_dict["cross_entropy_loss"]
        new_log = f"Validation {cfg['dataset_name']} | Epoch: {epoch+1}, " + ", ".join(
            [
                f"{loss_function}: {np.round(loss_value, 3)}"
                for loss_function, loss_value in val_loss_dict.items()
            ]
        )
        write_and_print_new_log(new_log, cfg)

        if cfg["use_lr_scheduler"]:
            scheduler.step(current_val_loss)

        if current_val_loss < best_val_loss:
            save_checkpoint(model, cfg)
            best_val_loss = current_val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter == cfg["early_stop_patience"]:
            break

    # Regular training done, unless we are using Autoencoder we just return
    # best model.
    if "ae" not in cfg["model_name"]:
        return load_checkpoint(cfg)

    # If we are using Autoencoder, load and train a classifier on encoded data.
    autoencoder = load_checkpoint(cfg)
    gbc = GradientBoostingClassifier(verbose=2)

    X, y, X_test, y_test = get_data(
        dataset_name=cfg["dataset_name"],
        dataset_dir=cfg["dataset_dir"],
        data_dim=187,
        seed=cfg["seed"],
    )

    def encode(X):
        X = torch.tensor(X, dtype=torch.float, device=cfg["device"])
        # Pad to large enough multiple of 2 so that no loss in dimensionality
        # through Autoencoder.
        X = pad_signals(X, 192)
        X_hat = autoencoder.encode(X)
        X_hat = torch.squeeze(X_hat)
        return X_hat.detach().cpu().numpy()

    X_hat = encode(X)
    X_test_hat = encode(X_test)

    gbc.fit(X_hat, y)

    # Save encoded data so it can be easily accessed for predictions.
    output_dir = os.path.join(get_checkpoints_dir(cfg), cfg["ae_output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    # We want to load data later in the same way we did initially, which
    # requires files for ptbdb to have suffixes "normal" and "abnormal".
    # Since at this point the two are mixed and when loaded will be mixed
    # again, the two file names have no special meaning and are purely for
    # consistency.
    train_suffix = "train" if cfg["dataset_name"] == "mitbih" else "normal"
    test_suffix = "test" if cfg["dataset_name"] == "mitbih" else "abnormal"

    train_filename = os.path.join(
        output_dir, f"{cfg['dataset_name']}_{train_suffix}.csv"
    )
    test_filename = os.path.join(output_dir, f"{cfg['dataset_name']}_{test_suffix}.csv")

    train_hat = np.hstack((X_hat, y.reshape((X_hat.shape[0], 1))))
    test_hat = np.hstack((X_test_hat, y_test.reshape((X_test_hat.shape[0], 1))))

    np.savetxt(train_filename, train_hat, delimiter=",")
    np.savetxt(test_filename, test_hat, delimiter=",")

    return gbc


def test(cfg, model, train_split, validation_split, test_split):
    if "ae" in cfg["model_name"]:
        seed = cfg["seed"]
        dataset_name = cfg["dataset_name"]
        val_ratio = 0.15 if dataset_name == "mitbih" else 0.2

        encoded_data_dir = os.path.join(get_checkpoints_dir(cfg), cfg["ae_output_dir"])
        X_train, y_train, X_test, y_test = get_data(
            dataset_name=dataset_name,
            dataset_dir=encoded_data_dir,
            data_dim=cfg["ae_latent_dim"],
            seed=seed,
        )

        X_train = np.squeeze(X_train)
        X_test = np.squeeze(X_test)

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_ratio, random_state=seed, stratify=y_train,
        )

        y_hat_train = model.predict_proba(X_train)
        y_hat_val = model.predict_proba(X_val)
        y_hat_test = model.predict_proba(X_test)

        if cfg["dataset_name"] == "ptbdb":
            y_hat_train = y_hat_train[:, 1:]
            y_hat_val = y_hat_val[:, 1:]
            y_hat_test = y_hat_test[:, 1:]

        train_data_loader = get_data_loader(cfg, split=train_split, shuffle=False)
        test_loss_dict = evaluate_predictions(
            y_test,
            y_hat_test,
            train_data_loader.dataset.class_weights,
            cfg,
            use_logits=False,
        )

        save_predictions_to_disk(y_train, y_hat_train, "train", cfg, use_logits=False)
        save_predictions_to_disk(y_val, y_hat_val, "val", cfg, use_logits=False)
        save_predictions_to_disk(y_test, y_hat_test, "test", cfg, use_logits=False)
    else:
        train_data_loader = get_data_loader(cfg, split=train_split, shuffle=False)
        val_data_loader = get_data_loader(cfg, split=validation_split, shuffle=False)
        test_data_loader = get_data_loader(cfg, split=test_split, shuffle=False)

        class_weights = train_data_loader.dataset.class_weights

        evaluation_epoch(
            model,
            train_data_loader,
            class_weights,
            train_split,
            cfg,
            save_to_disk=True,
        )
        evaluation_epoch(
            model,
            val_data_loader,
            class_weights,
            validation_split,
            cfg,
            save_to_disk=True,
        )
        test_loss_dict = evaluation_epoch(
            model, test_data_loader, class_weights, test_split, cfg, save_to_disk=True,
        )

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
    cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seeds(cfg)

    write_and_print_new_log(
        f"Dataset name: {cfg['dataset_name']}, Model name: {cfg['model_name']}, Transfer learning: {cfg['transfer_learning']}, RNN freeze: {cfg['rnn_freeze']}, RNN Bidirectional: {cfg['rnn_bidirectional']}, RNN Num Layers: {cfg['rnn_num_layers']}",
        cfg,
    )

    if not cfg["transfer_learning"]:  # task 1 or 2
        model = train(cfg, model=None, train_split="train", validation_split="val",)
        test(
            cfg,
            model=model,
            train_split="train",
            validation_split="val",
            test_split="test",
        )
    else:  # task 4

        # train on mitbih
        cfg["dataset_name"] = "mitbih"
        assert (
            "rnn" in cfg["model_name"]
        ), "Transfer learning task was only implemented for RNN."
        model = train(
            cfg, model=None, train_split="train_val", validation_split="test",
        )

        # freeze all weights if necessary
        if cfg["rnn_freeze"] in ["permanent", "temporary"]:
            write_and_print_new_log("Freezing weights...", cfg)
            for param in model.parameters():
                param.requires_grad = False
        elif cfg["rnn_freeze"] != "never":
            raise Exception(f"Not a valid rnn_freeze {cfg['rnn_freeze']}.")

        # replace FCNN with a suitable one. newly added layer's weights have requires_grad = True by default
        model.fc = nn.Linear(model.rnn_output_size, 1, device=cfg["device"])

        # train and test on ptbdb
        cfg["dataset_name"] = "ptbdb"
        model = train(cfg, model=model, train_split="train", validation_split="val",)
        test(
            cfg,
            model=model,
            train_split="train",
            validation_split="val",
            test_split="test",
        )
