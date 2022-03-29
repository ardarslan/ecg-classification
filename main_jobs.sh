cd src

# Vanilla SimpleRNN for MIT-BIH dataset
python main.py --dataset_name mitbih --model_name vanilla_rnn

# Vanilla LSTM for MIT-BIH dataset
python main.py --dataset_name mitbih --model_name lstm_rnn

# Bidirectional RNN for MIT-BIH dataset
python main.py --dataset_name mitbih --model_name vanilla_rnn --rnn_bidirectional

# Bidirectional LSTM for MIT-BIH dataset
python main.py --dataset_name mitbih --model_name lstm_rnn --rnn_bidirectional

# Vanilla CNN for MIT-BIH dataset
python main.py --dataset_name mitbih --model_name vanilla_cnn

# Residual CNN for MIT-BIH dataset
python main.py --dataset_name mitbih --model_name residual_cnn

# Autoencoder + GBC for MIT-BIH dataset
python main.py --dataset_name mitbih --model_name ae

# 1D Inception Net for MIT-BIH dataset
python main.py --dataset_name mitbih --model_name inception_net

# Shared MLP Over Vanilla RNN for MIT-BIH dataset
python main.py --dataset_name mitbih --model_name sharedmlpover_vanilla_rnn

# Shared MLP Over LSTM for MIT-BIH dataset
python main.py --dataset_name mitbih --model_name sharedmlpover_lstm_rnn

# Vanilla SimpleRNN for PTBDB dataset
python main.py --dataset_name ptbdb --model_name vanilla_rnn

# Vanilla LSTM for PTBDB dataset
python main.py --dataset_name ptbdb --model_name lstm_rnn

# Bidirectional RNN for PTBDB dataset
python main.py --dataset_name ptbdb --model_name vanilla_rnn --rnn_bidirectional

# Bidirectional LSTM for PTBDB dataset
python main.py --dataset_name ptbdb --model_name lstm_rnn --rnn_bidirectional

# Vanilla CNN for PTBDB dataset
python main.py --dataset_name ptbdb --model_name vanilla_cnn

# Residual CNN for PTBDB dataset
python main.py --dataset_name ptbdb --model_name residual_cnn

# Autoencoder + GBC for PTBDB dataset
python main.py --dataset_name ptbdb --model_name ae

# 1D Inception Net for PTBDB dataset
python main.py --dataset_name ptbdb --model_name inception_net

# Shared MLP Over SimpleRNN for PTBDB dataset
python main.py --dataset_name ptbdb --model_name sharedmlpover_vanilla_rnn

# Shared MLP Over LSTM for PTBDB dataset
python main.py --dataset_name ptbdb --model_name sharedmlpover_lstm_rnn

# Transfer Learning (SimpleRNN, PF)
python main.py --dataset_name ptbdb --model_name vanilla_rnn --transfer_learning --rnn_freeze permanent

# Transfer Learning (LSTM, PF)
python main.py --dataset_name ptbdb --model_name lstm_rnn --transfer_learning --rnn_freeze permanent

# Transfer Learning (Bidirectional SimpleRNN, PF)
python main.py --dataset_name ptbdb --model_name vanilla_rnn --rnn_bidirectional --transfer_learning --rnn_freeze permanent

# Transfer Learning (Bidirectional LSTM, PF)
python main.py --dataset_name ptbdb --model_name lstm_rnn --rnn_bidirectional --transfer_learning --rnn_freeze permanent

# Transfer Learning (SimpleRNN, TF)
python main.py --dataset_name ptbdb --model_name vanilla_rnn --transfer_learning --rnn_freeze temporary

# Transfer Learning (LSTM, TF)
python main.py --dataset_name ptbdb --model_name lstm_rnn --transfer_learning --rnn_freeze temporary

# Transfer Learning (Bidirectional SimpleRNN, TF)
python main.py --dataset_name ptbdb --model_name vanilla_rnn --rnn_bidirectional --transfer_learning --rnn_freeze temporary

# Transfer Learning (Bidirectional LSTM, TF)
python main.py --dataset_name ptbdb --model_name lstm_rnn --rnn_bidirectional --transfer_learning --rnn_freeze temporary

# Transfer Learning (SimpleRNN, NF)
python main.py --dataset_name ptbdb --model_name vanilla_rnn --transfer_learning --rnn_freeze never

# Transfer Learning (LSTM, NF)
python main.py --dataset_name ptbdb --model_name lstm_rnn --transfer_learning --rnn_freeze never

# Transfer Learning (Bidirectional SimpleRNN, NF)
python main.py --dataset_name ptbdb --model_name vanilla_rnn --rnn_bidirectional --transfer_learning --rnn_freeze never

# Transfer Learning (Bidirectional LSTM, NF)
python main.py --dataset_name ptbdb --model_name lstm_rnn --rnn_bidirectional --transfer_learning --rnn_freeze never
