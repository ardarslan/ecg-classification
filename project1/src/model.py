import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if cfg["rnn_type"] == "vanilla_rnn":
            self.rnn = nn.RNN
        elif cfg["rnn_type"] == "lstm_rnn":
            self.rnn = nn.LSTM
        elif cfg["rnn_type"] == "gru_rnn":
            self.rnn = nn.GRU
        else:
            raise Exception(f"Not a valid rnn_type {cfg['rnn_type']}.")

        self.rnn = self.rnn(input_size=1,
                            hidden_size=cfg["rnn_hidden_size"],
                            num_layers=cfg["rnn_num_layers"],
                            bidirectional=cfg["rnn_bidirectional"],
                            dropout=cfg["rnn_dropout"],
                            batch_first=True)
        rnn_output_size = cfg["rnn_hidden_size"]
        if cfg["rnn_bidirectional"]:
            rnn_output_size *= 2
        if cfg["dataset_name"] == "ptbdb":
            linear_output_size = 1
        elif cfg["dataset_name"] == "mitbih":
            linear_output_size = 5
        else:
            raise Exception(f"Not a valid dataset_name {cfg['dataset_name']}.")
        self.fc = nn.Linear(rnn_output_size, linear_output_size)

    def forward(self, X):
        _, last_cell_hidden_states = self.rnn.forward(X) # (D*n_layers, N, H_out)
        last_cell_hidden_states = last_cell_hidden_states.view(self.cfg["rnn_num_layers"], 1 + 1 * self.cfg["rnn_bidirectional"], -1, self.cfg["rnn_hidden_size"])
        last_cell_hidden_states_of_last_layer = last_cell_hidden_states[-1, :, :, :] # (D, N, H_out)
        last_cell_hidden_states_of_last_layer = last_cell_hidden_states_of_last_layer.permute(1, 0, 2).contiguous().view(self.cfg["batch_size"], -1) # (N, D*H_out)
        output = self.fc(last_cell_hidden_states_of_last_layer)
        return output


class CNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        # current_num_features = 187
        # previous_num_features = None
        for i in range(cfg["cnn_num_layers"]):
            if i == 0:
                in_channels = 1
            else:
                in_channels = cfg["cnn_num_channels"]

            self.conv_layers.append(nn.Conv1d(in_channels=in_channels, out_channels=cfg["cnn_num_channels"], kernel_size=3, stride=1, bias=False, padding="same"))
            # previous_num_features = current_num_features
            # current_num_features = int(np.floor((current_num_features - 3) / 2 + 1)) # due to conv
            # current_num_features = int(np.floor((current_num_features - 2) / 2 + 1)) # due to max_pool
            self.batch_norms.append(nn.BatchNorm1d(num_features=cfg["cnn_num_channels"]))
        self.avg_pool = nn.AvgPool1d(kernel_size=187) # global average pooling
        self.fc1 = nn.Linear(cfg["cnn_num_channels"], 32)
        if cfg["dataset_name"] == "ptbdb":
            linear_output_size = 1
        elif cfg["dataset_name"] == "mitbih":
            linear_output_size = 5
        else:
            raise Exception(f"Not a valid dataset_name {cfg['dataset_name']}")
        self.fc2 = nn.Linear(32, linear_output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1) # (N, L, C) -> (N, C, L)
        for i in range(self.cfg["cnn_num_layers"]):
            if self.cfg["cnn_type"] == "vanilla_cnn":
                x = F.leaky_relu(self.batch_norms[i](self.conv_layers[i](x)))
            elif self.cfg["cnn_type"] == "residual_cnn":
                if i == 0:
                    x = x.repeat(1, self.cfg["cnn_num_channels"], 1) + F.leaky_relu(self.batch_norms[i](self.conv_layers[i](x)))
                else:
                    x = x + F.leaky_relu(self.batch_norms[i](self.conv_layers[i](x)))
            else:
                raise Exception(f"Not a valid cnn_type {self.cfg['cnn_type']}.")
        x = self.avg_pool(x).squeeze()
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x) # (N, num_classes) -> logits
        return x
