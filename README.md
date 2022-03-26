# Mitbih


## BiDirectional RNN:

bsub -n 2 -W 24:00 -R "rusage[mem=8192]" python main.py --dataset_name mitbih --model_name vanilla_rnn

## BiDirectional LSTM:

bsub -n 2 -W 24:00 -R "rusage[mem=8192]" python main.py --dataset_name mitbih --model_name lstm_rnn

## Vanilla CNN:

bsub -n 2 -W 24:00 -R "rusage[mem=8192]" python main.py --dataset_name mitbih --model_name vanilla_cnn

## Residual CNN:

bsub -n 2 -W 24:00 -R "rusage[mem=8192]" python main.py --dataset_name mitbih --model_name residual_cnn

## Majority Vote Ensemble

```
python ensemble.py --dataset_name mitbih --model_name majority_ensemble
```

## Logistic Regression Ensemble

```
python ensemble.py --dataset_name mitbih --model_name log_reg_ensemble
```

# Ptbdb

## BiDirectional RNN:

bsub -n 2 -W 24:00 -R "rusage[mem=8192]" python main.py --dataset_name ptbdb --model_name vanilla_rnn

## BiDirectional LSTM:

bsub -n 2 -W 24:00 -R "rusage[mem=8192]" python main.py --dataset_name ptbdb --model_name lstm_rnn

## Vanilla CNN:

bsub -n 2 -W 24:00 -R "rusage[mem=8192]" python main.py --dataset_name ptbdb --model_name vanilla_cnn

## Residual CNN:

bsub -n 2 -W 24:00 -R "rusage[mem=8192]" python main.py --dataset_name ptbdb --model_name residual_cnn

## Majority Vote Ensemble

```
python ensemble.py --dataset_name ptbdb --model_name majority_ensemble
```

## Logistic Regression Ensemble

```
python ensemble.py --dataset_name ptbdb --model_name log_reg_ensemble
```


Please see checkpoints folder for results.