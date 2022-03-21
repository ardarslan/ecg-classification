# Mitbih


## BiDirectional RNN:

bsub -n 2 -W 24:00 -R "rusage[mem=8192]" python main.py --dataset_name mitbih --model_name vanilla_rnn

## BiDirectional LSTM:

bsub -n 2 -W 24:00 -R "rusage[mem=8192]" python main.py --dataset_name mitbih --model_name lstm_rnn

## Vanilla CNN:

bsub -n 2 -W 24:00 -R "rusage[mem=8192]" python main.py --dataset_name mitbih --model_name vanilla_cnn

## Residual CNN:

bsub -n 2 -W 24:00 -R "rusage[mem=8192]" python main.py --dataset_name mitbih --model_name residual_cnn


# Ptbdb

## BiDirectional RNN:

bsub -n 2 -W 24:00 -R "rusage[mem=8192]" python main.py --dataset_name ptbdb --model_name vanilla_rnn

## BiDirectional LSTM:

bsub -n 2 -W 24:00 -R "rusage[mem=8192]" python main.py --dataset_name ptbdb --model_name lstm_rnn

## Vanilla CNN:

bsub -n 2 -W 24:00 -R "rusage[mem=8192]" python main.py --dataset_name ptbdb --model_name vanilla_cnn

## Residual CNN:

bsub -n 2 -W 24:00 -R "rusage[mem=8192]" python main.py --dataset_name ptbdb --model_name residual_cnn



Please see checkpoints folder for results.