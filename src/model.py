import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if cfg["model_name"] == "vanilla_rnn":
            self.rnn = nn.RNN
        elif cfg["model_name"] == "lstm_rnn":
            self.rnn = nn.LSTM
        elif cfg["model_name"] == "gru_rnn":
            self.rnn = nn.GRU
        else:
            raise Exception(f"Not a valid model_name {cfg['model_name']}.")
        self.D = 1 + 1 * self.cfg["rnn_bidirectional"]
        self.rnn = self.rnn(
            input_size=1,
            hidden_size=cfg["rnn_hidden_size"],
            num_layers=cfg["rnn_num_layers"],
            bidirectional=cfg["rnn_bidirectional"],
            dropout=cfg["rnn_dropout"],
            batch_first=True,
        )
        self.rnn_output_size = self.D * cfg["rnn_hidden_size"]
        if cfg["dataset_name"] == "ptbdb":
            linear_output_size = 1
        elif cfg["dataset_name"] == "mitbih":
            linear_output_size = 5
        else:
            raise Exception(f"Not a valid dataset_name {cfg['dataset_name']}.")
        self.fc = nn.Linear(self.rnn_output_size, linear_output_size)

    def forward(self, X):
        N, L, _ = X.shape
        if self.cfg["model_name"] == "lstm_rnn":
            _, last_cell_hidden_states_h_and_c = self.rnn.forward(X)
            last_cell_hidden_states = last_cell_hidden_states_h_and_c[
                0
            ]  # (D*n_layers, N, H_out)
        elif self.cfg["model_name"] in ["gru_rnn", "vanilla_rnn"]:
            _, last_cell_hidden_states = self.rnn.forward(X)  # (D*n_layers, N, H_out)
        else:
            raise Exception(f"Not a valid model_name {self.cfg['model_name']}.")
        last_cell_hidden_states = F.leaky_relu(last_cell_hidden_states)
        last_cell_hidden_states = last_cell_hidden_states.view(
            self.cfg["rnn_num_layers"], self.D, N, self.cfg["rnn_hidden_size"]
        )
        last_cell_hidden_states_of_last_layer = last_cell_hidden_states[
            -1, :, :, :
        ]  # (D, N, H_out)
        last_cell_hidden_states_of_last_layer = (
            last_cell_hidden_states_of_last_layer.permute(1, 0, 2)
            .contiguous()
            .view(N, self.rnn_output_size)
        )  # (N, D*H_out)
        output = self.fc(last_cell_hidden_states_of_last_layer)  # (N, num_logits)
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

            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=cfg["cnn_num_channels"],
                    kernel_size=3,
                    stride=1,
                    bias=False,
                    padding="same",
                )
            )
            # previous_num_features = current_num_features
            # current_num_features = int(np.floor((current_num_features - 3) / 2 + 1)) # due to conv
            # current_num_features = int(np.floor((current_num_features - 2) / 2 + 1)) # due to max_pool
            self.batch_norms.append(
                nn.BatchNorm1d(num_features=cfg["cnn_num_channels"])
            )
        self.avg_pool = nn.AvgPool1d(kernel_size=187)  # global average pooling
        self.fc1 = nn.Linear(cfg["cnn_num_channels"], 32)
        if cfg["dataset_name"] == "ptbdb":
            linear_output_size = 1
        elif cfg["dataset_name"] == "mitbih":
            linear_output_size = 5
        else:
            raise Exception(f"Not a valid dataset_name {cfg['dataset_name']}")
        self.fc2 = nn.Linear(32, linear_output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (N, L, C) -> (N, C, L)
        for i in range(self.cfg["cnn_num_layers"]):
            if self.cfg["model_name"] == "vanilla_cnn":
                x = F.leaky_relu(self.batch_norms[i](self.conv_layers[i](x)))
            elif self.cfg["model_name"] == "residual_cnn":
                if i == 0:
                    x = x.repeat(1, self.cfg["cnn_num_channels"], 1) + F.leaky_relu(
                        self.batch_norms[i](self.conv_layers[i](x))
                    )
                else:
                    x = x + F.leaky_relu(self.batch_norms[i](self.conv_layers[i](x)))
            else:
                raise Exception(f"Not a valid model_name {self.cfg['model_name']}.")
        x = self.avg_pool(x).squeeze()
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)  # (N, num_logits) -> logits
        return x


class Autoencoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        latent_dim = cfg["ae_latent_dim"]

        ## Encoder ##
        conv = nn.Conv1d(
            in_channels=1, out_channels=5, kernel_size=3, stride=2, padding=1
        )
        max_pool = nn.MaxPool1d(kernel_size=2)
        conv2 = nn.Conv1d(
            in_channels=5, out_channels=15, kernel_size=3, stride=2, padding=1
        )
        max_pool2 = nn.MaxPool1d(kernel_size=2)
        conv3 = nn.Conv1d(
            in_channels=15, out_channels=30, kernel_size=3, stride=2, padding=1
        )
        max_pool3 = nn.MaxPool1d(kernel_size=2)
        flatten = nn.Flatten()
        dense = nn.Linear(in_features=90, out_features=latent_dim)

        self.encoder = nn.Sequential(
            conv,
            nn.PReLU(),
            nn.BatchNorm1d(num_features=5),
            max_pool,
            conv2,
            nn.PReLU(),
            nn.BatchNorm1d(num_features=15),
            max_pool2,
            conv3,
            nn.PReLU(),
            nn.BatchNorm1d(num_features=30),
            max_pool3,
            flatten,
            dense,
            nn.Dropout(p=0.1),
        )

        ## Decoder ##
        dec_dense = nn.Linear(in_features=latent_dim, out_features=90)
        reshape = nn.Unflatten(dim=1, unflattened_size=(30, 3))
        convt3 = nn.ConvTranspose1d(
            in_channels=30,
            out_channels=30,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        upsample3 = nn.Upsample(scale_factor=2)
        convt2 = nn.ConvTranspose1d(
            in_channels=30,
            out_channels=15,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        upsample2 = nn.Upsample(scale_factor=2)
        convt = nn.ConvTranspose1d(
            in_channels=15,
            out_channels=5,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        upsample = nn.Upsample(scale_factor=2)
        convt_final = nn.ConvTranspose1d(
            in_channels=5, out_channels=1, kernel_size=3, stride=1, padding=1
        )

        self.decoder = nn.Sequential(
            dec_dense,
            nn.PReLU(),
            nn.Dropout(p=0.1),
            nn.BatchNorm1d(num_features=90),
            reshape,
            convt3,
            nn.PReLU(),
            nn.BatchNorm1d(num_features=30),
            upsample3,
            convt2,
            nn.PReLU(),
            nn.BatchNorm1d(num_features=15),
            upsample2,
            convt,
            nn.PReLU(),
            nn.BatchNorm1d(num_features=5),
            upsample,
            convt_final,
        )

    def encode(self, x):
        x = x.permute(0, 2, 1)  # (N, L, C) -> (N, C, L)
        return self.encoder(x)

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decoder(encoded)
        return decoded
