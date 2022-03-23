import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from utils import set_seeds, get_model, get_optimizer, get_scheduler, \
                  get_data_loader, save_checkpoint, load_checkpoint, \
                  evaluate_predictions, write_and_print_new_log, \
                  save_predictions_to_disk


def train_epoch(model, optimizer, train_data_loader, class_weights, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    all_y = []
    all_yhat = []
    for batch in train_data_loader:
        optimizer.zero_grad()
        X, y = batch["X"].float().to(device), batch["y"].long().to(device)
        yhat = model(X)
        if cfg["dataset_name"] == "mitbih":
            cross_entropy_loss = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights))(yhat, y)
        elif cfg["dataset_name"] == "ptbdb":
            sample_weights = np.array([class_weights[int(label)] for label in y], dtype=np.float32)
            cross_entropy_loss = torch.nn.BCEWithLogitsLoss(weight=torch.tensor(sample_weights))(yhat.squeeze(), y.float())
        else:
            raise Exception(f"Not a valid dataset {cfg['dataset_name']}.")
        all_y.append(y.detach().cpu().numpy())
        all_yhat.append(yhat.detach().cpu().numpy())
        cross_entropy_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_max_norm"])
        optimizer.step()
    all_y = np.concatenate(all_y, axis=0)
    all_yhat = np.concatenate(all_yhat, axis=0).astype(np.float32)
    train_loss_dict = evaluate_predictions(all_y, all_yhat, class_weights, cfg)
    return train_loss_dict


def evaluation_epoch(model, evaluation_data_loader, class_weights, split, cfg, save_to_disk=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        all_y = []
        all_yhat = []
        for batch in evaluation_data_loader:
            X, y = batch["X"].float().to(device), batch["y"].long().to(device)
            yhat = model(X)
            all_y.append(y.detach().cpu().numpy())
            all_yhat.append(yhat.detach().cpu().numpy())
        all_y = np.concatenate(all_y, axis=0)
        all_yhat = np.concatenate(all_yhat, axis=0).astype(np.float32)
        if save_to_disk:
            save_predictions_to_disk(all_y, all_yhat, split, cfg)
        eval_loss_dict = evaluate_predictions(all_y, all_yhat, class_weights, cfg)
    return eval_loss_dict


def train(cfg, model, train_split, validation_split, test_split):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seeds(cfg)

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
        # if we are in the second training part of transfer learning task and the rule is to unfreeze RNN after rnn_freeze_num_epochs:
        if cfg["transfer_learning"] and cfg["dataset_name"] == "ptbdb" and cfg["rnn_freeze"] == "temporary" and cfg["rnn_freeze_num_epochs"] == epoch:
            write_and_print_new_log("Unfreezing weights...", cfg)
            for param in model.parameters():
                param.requires_grad = True

        # train
        train_loss_dict = train_epoch(model, optimizer, train_data_loader, class_weights, cfg)
        new_log = f"Train {cfg['dataset_name']} | Epoch: {epoch+1}, " + ", ".join([f"{loss_function}: {np.round(loss_value, 3)}" for loss_function, loss_value in train_loss_dict.items()])
        write_and_print_new_log(new_log, cfg)

        # validate
        val_loss_dict = evaluation_epoch(model, val_data_loader, class_weights, "val", cfg, save_to_disk=False)
        current_val_loss = val_loss_dict['cross_entropy_loss']
        new_log = f"Validation {cfg['dataset_name']} | Epoch: {epoch+1}, " + ", ".join([f"{loss_function}: {np.round(loss_value, 3)}" for loss_function, loss_value in val_loss_dict.items()])
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
    
    if test_split:
        model = load_checkpoint(cfg)
        
        train_data_loader = get_data_loader(cfg, split=train_split, shuffle=False)
        val_data_loader = get_data_loader(cfg, split=validation_split, shuffle=False)
        test_data_loader = get_data_loader(cfg, split=test_split, shuffle=False)

        evaluation_epoch(model, train_data_loader, class_weights, train_split, cfg, save_to_disk=True)
        evaluation_epoch(model, val_data_loader, class_weights, validation_split, cfg, save_to_disk=True)
        test_loss_dict = evaluation_epoch(model, test_data_loader, class_weights, test_split, cfg, save_to_disk=True)

        new_log = f"Test {cfg['dataset_name']} | " + ", ".join([f"{loss_function}: {np.round(loss_value, 3)}" for loss_function, loss_value in test_loss_dict.items()])
        write_and_print_new_log(new_log, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for running the script')

    parser.add_argument('--dataset_dir', type=str, default='../data')
    parser.add_argument('--checkpoints_dir', type=str, default='../checkpoints')
    parser.add_argument('--dataset_name', type=str, default='mitbih')  # mitbih, ptbdb
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=1)  # 0 means use the same thread for data processing
    parser.add_argument('--model_name', type=str, default='vanilla_rnn')  # vanilla_rnn, lstm_rnn, gru_rnn, vanilla_cnn, residual_cnn
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--lr', type=int, default=0.001)
    parser.add_argument('--weight_decay', type=int, default=0.0)
    parser.add_argument('--use_lr_scheduler', type=int, default=True)
    parser.add_argument('--lr_scheduler_patience', type=int, default=5)
    parser.add_argument('--early_stop_patience', type=int, default=15)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--gradient_max_norm', type=int, default=5.0)
    parser.add_argument('--transfer_learning', action='store_true')

    # rnn configs
    parser.add_argument('--rnn_hidden_size', type=int, default=128)
    parser.add_argument('--rnn_num_layers', type=int, default=1)
    parser.add_argument('--rnn_bidirectional', action='store_true')
    parser.add_argument('--rnn_dropout', type=float, default=0.0)
    parser.add_argument('--rnn_freeze', type=str, default="never", help=""" - permanent: train only a new FCNN on top of RNN, """
                                                                        """ - temporary: train only a new FCNN on top of RNN """
                                                                        """ for 'rnn_freeze_num_epochs', after that start training the """
                                                                        """ RNN as well, """
                                                                        """ - never: both RNN and FCNN will be trained from the """
                                                                        """ the beginning of finetuning.""")
    parser.add_argument('--rnn_freeze_num_epochs', type=int, default=20)

    # cnn configs
    parser.add_argument('--cnn_num_layers', type=int, default=4)
    parser.add_argument('--cnn_num_channels', type=int, default=64)

    cfg = parser.parse_args().__dict__
    cfg["experiment_time"] = str(int(time.time()))

    write_and_print_new_log(f"Dataset name: {cfg['dataset_name']}, Model name: {cfg['model_name']}, Transfer learning: {cfg['transfer_learning']}, RNN freeze: {cfg['rnn_freeze']}, RNN Bidirectional: {cfg['rnn_bidirectional']}, RNN Num Layers: {cfg['rnn_num_layers']}", cfg)

    if not cfg["transfer_learning"]: # task 1 or 2
        train(cfg, model=None, train_split="train", validation_split="val", test_split="test")
    else: # task 4

        # train on mitbih
        cfg["dataset_name"] = "mitbih"
        assert "rnn" in cfg["model_name"], "Transfer learning task was only implemented for RNN."
        train(cfg, model=None, train_split="train_val", validation_split="test", test_split=None)
        model = load_checkpoint(cfg)
        
        # freeze all weights if necessary
        if cfg["rnn_freeze"] in ["permanent", "temporary"]:
            write_and_print_new_log("Freezing weights...", cfg)
            for param in model.parameters():
                param.requires_grad = False
        elif cfg["rnn_freeze"] == "never":
            pass
        else:
            raise Exception(f"Not a valid rnn_freeze {cfg['rnn_freeze']}.")
        
        # replace FCNN with a suitable one. newly added layer's weights have requires_grad = True by default
        model.fc = nn.Linear(model.rnn_output_size, 1)

        # train and test on ptbdb
        cfg["dataset_name"] = "ptbdb"
        train(cfg, model=model, train_split="train", validation_split="val", test_split="test")
