import numpy as np
import torch
import torch.nn as nn
from utils import set_seeds, get_model, get_optimizer, get_scheduler, \
                  get_data_loader, save_checkpoint, load_checkpoint, \
                  evaluate_predictions, read_config


def train_epoch(model, optimizer, train_data_loader, class_weights, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    all_y = []
    all_yhat = []
    for batch in train_data_loader:
        optimizer.zero_grad()
        X, y = batch["X"].float().to(device), batch["y"].long().to(device)
        sample_weights = np.array([class_weights[int(label)] for label in y], dtype=np.float32)
        yhat = model(X)
        if cfg["dataset_name"] == "mitbih":
            cross_entropy_loss = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights))(yhat, y)
        elif cfg["dataset_name"] == "ptbdb":
            cross_entropy_loss = torch.nn.BCEWithLogitsLoss(weight=torch.tensor(sample_weights).unsqueeze(-1))(yhat, y.float())
        else:
            raise Exception(f"Not a valid dataset {cfg['dataset_name']}.")
        all_y.append(y.detach().cpu())
        all_yhat.append(yhat.detach().cpu())
        cross_entropy_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_max_norm"])
        optimizer.step()
    all_y = np.concatenate(all_y, axis=0)
    all_yhat = np.concatenate(all_yhat, axis=0).astype(np.float32)
    train_loss_dict = evaluate_predictions(all_y, all_yhat, class_weights, cfg)
    return train_loss_dict


def evaluation_epoch(model, evaluation_data_loader, class_weights, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        all_y = []
        all_yhat = []
        for batch in evaluation_data_loader:
            X, y = batch["X"].float().to(device), batch["y"].long().to(device)
            yhat = model(X)
            all_y.append(y.detach().cpu())
            all_yhat.append(yhat.detach().cpu())
        all_y = np.concatenate(all_y, axis=0)
        all_yhat = np.concatenate(all_yhat, axis=0).astype(np.float32)
        eval_loss_dict = evaluate_predictions(all_y, all_yhat, class_weights, cfg)
    return eval_loss_dict


def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seeds(cfg)

    model = get_model(cfg).to(device)
    optimizer = get_optimizer(cfg, model)

    if cfg["use_lr_scheduler"]:
        scheduler = get_scheduler(cfg, optimizer)

    train_data_loader = get_data_loader(cfg, split="train")
    class_weights = train_data_loader.dataset.class_weights
    val_data_loader = get_data_loader(cfg, split="val")
    test_data_loader = get_data_loader(cfg, split="test")

    best_val_loss = np.inf
    early_stop_counter = 0
    for epoch in range(cfg["max_epochs"]):
        # train
        train_loss_dict = train_epoch(model, optimizer, train_data_loader, class_weights, cfg)
        print(f"Train | Epoch: {epoch+1}, " + ", ".join([f"{loss_function}: {np.round(loss_value, 3)}" for loss_function, loss_value in train_loss_dict.items()]))

        # validate
        val_loss_dict = evaluation_epoch(model, val_data_loader, class_weights, cfg)
        current_val_loss = val_loss_dict['cross_entropy_loss']
        print(f"Validation | Epoch: {epoch+1}, " + ", ".join([f"{loss_function}: {np.round(loss_value, 3)}" for loss_function, loss_value in val_loss_dict.items()]))

        if cfg["use_lr_scheduler"]:
            scheduler.step(current_val_loss)

        if current_val_loss < best_val_loss:
            save_checkpoint(model, optimizer, train_loss_dict["cross_entropy_loss"], epoch, cfg)
            best_val_loss = current_val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter == cfg["early_stop_patience"]:
            break

    # test
    # train_val_data_loader = get_data_loader(cfg, split="train_val")
    # class_weights = train_val_data_loader.dataset.class_weights
    # val_data_loader = get_data_loader(cfg, split="val")
    # test_data_loader = get_data_loader(cfg, split="test")
    # model, optimizer, train_loss_at_early_stop, epoch = load_checkpoint(cfg)

    # Algorithm 7.3 in https://www.deeplearningbook.org/contents/regularization.html
    # got_lower_val_loss = False
    # for epoch_idx in range(cfg["max_epochs_before_test"]):
    #     train_val_loss_dict = train_epoch(model, optimizer, train_val_data_loader, class_weights, cfg)
    #     print(f"Train+Val | Epoch: {epoch+epoch_idx+1}, " + ", ".join([f"{loss_function}: {np.round(loss_value, 3)}" for loss_function, loss_value in train_val_loss_dict.items()]))
    #     val_loss_dict = evaluation_epoch(model, val_data_loader, class_weights, cfg)
    #     print(f"Val | Epoch: {epoch+epoch_idx+1}, " + ", ".join([f"{loss_function}: {np.round(loss_value, 3)}" for loss_function, loss_value in val_loss_dict.items()]))
    #     current_val_loss = val_loss_dict["cross_entropy_loss"]
    #     if current_val_loss < train_loss_at_early_stop:
    #         got_lower_val_loss = True
    #         break
    
    # if not got_lower_val_loss:
    model, _, _, _ = load_checkpoint(cfg)

    test_loss_dict = evaluation_epoch(model, test_data_loader, class_weights, cfg)
    print("Test | " + ", ".join([f"{loss_function}: {np.round(loss_value, 3)}" for loss_function, loss_value in test_loss_dict.items()]))


if __name__ == "__main__":
    cfg = read_config("../configs/config.yaml")
    train(cfg)
