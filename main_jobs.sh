cd src

# Vanilla SimpleRNN for MIT-BIH dataset
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name mitbih --model_name vanilla_rnn

# Vanilla LSTM for MIT-BIH dataset
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name mitbih --model_name lstm_rnn

# Bidirectional RNN for MIT-BIH dataset
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name mitbih --model_name vanilla_rnn --rnn_bidirectional

# Bidirectional LSTM for MIT-BIH dataset
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name mitbih --model_name lstm_rnn --rnn_bidirectional

# Vanilla CNN for MIT-BIH dataset
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name mitbih --model_name vanilla_cnn

# Residual CNN for MIT-BIH dataset
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name mitbih --model_name residual_cnn

# Autoencoder + GBC for MIT-BIH dataset
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name mitbih --model_name ae

# 1D Inception Net for MIT-BIH dataset
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name mitbih --model_name inception_net

# Shared MLP Over Vanilla RNN for MIT-BIH dataset
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name mitbih --model_name sharedmlpover_vanilla_rnn

# Shared MLP Over LSTM for MIT-BIH dataset
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name mitbih --model_name sharedmlpover_lstm_rnn

# Vanilla SimpleRNN for PTBDB dataset
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name ptbdb --model_name vanilla_rnn

# Vanilla LSTM for PTBDB dataset
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name ptbdb --model_name lstm_rnn

# Bidirectional RNN for PTBDB dataset
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name ptbdb --model_name vanilla_rnn --rnn_bidirectional

# Bidirectional LSTM for PTBDB dataset
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name ptbdb --model_name lstm_rnn --rnn_bidirectional

# Vanilla CNN for PTBDB dataset
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name ptbdb --model_name vanilla_cnn

# Residual CNN for PTBDB dataset
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name ptbdb --model_name residual_cnn

# Autoencoder + GBC for PTBDB dataset
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name ptbdb --model_name ae

# 1D Inception Net for PTBDB dataset
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name ptbdb --model_name inception_net

# Shared MLP Over SimpleRNN for PTBDB dataset
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name ptbdb --model_name sharedmlpover_vanilla_rnn

# Shared MLP Over LSTM for PTBDB dataset
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name ptbdb --model_name sharedmlpover_lstm_rnn

# Transfer Learning (SimpleRNN, PF)
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name ptbdb --model_name vanilla_rnn --transfer_learning --rnn_freeze permanent

# Transfer Learning (LSTM, PF)
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name ptbdb --model_name lstm_rnn --transfer_learning --rnn_freeze permanent

# Transfer Learning (Bidirectional SimpleRNN, PF)
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name ptbdb --model_name vanilla_rnn --rnn_bidirectional --transfer_learning --rnn_freeze permanent

# Transfer Learning (Bidirectional LSTM, PF)
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name ptbdb --model_name lstm_rnn --rnn_bidirectional --transfer_learning --rnn_freeze permanent

# Transfer Learning (SimpleRNN, TF)
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name ptbdb --model_name vanilla_rnn --transfer_learning --rnn_freeze temporary

# Transfer Learning (LSTM, TF)
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name ptbdb --model_name lstm_rnn --transfer_learning --rnn_freeze temporary

# Transfer Learning (Bidirectional SimpleRNN, TF)
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name ptbdb --model_name vanilla_rnn --rnn_bidirectional --transfer_learning --rnn_freeze temporary

# Transfer Learning (Bidirectional LSTM, TF)
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name ptbdb --model_name lstm_rnn --rnn_bidirectional --transfer_learning --rnn_freeze temporary

# Transfer Learning (SimpleRNN, NF)
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name ptbdb --model_name vanilla_rnn --transfer_learning --rnn_freeze never

# Transfer Learning (LSTM, NF)
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name ptbdb --model_name lstm_rnn --transfer_learning --rnn_freeze never

# Transfer Learning (Bidirectional SimpleRNN, NF)
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name ptbdb --model_name vanilla_rnn --rnn_bidirectional --transfer_learning --rnn_freeze never

# Transfer Learning (Bidirectional LSTM, NF)
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name ptbdb --model_name lstm_rnn --rnn_bidirectional --transfer_learning --rnn_freeze never
