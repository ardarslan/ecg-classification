bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name mitbih --model_name vanilla_rnn
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name mitbih --model_name lstm_rnn
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name mitbih --model_name vanilla_rnn --rnn_bidirectional
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name mitbih --model_name lstm_rnn --rnn_bidirectional
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name mitbih --model_name vanilla_cnn
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name mitbih --model_name residual_cnn
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name mitbih --model_name ae
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name mitbih --model_name inception_net
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name mitbih --model_name attention_vanilla_rnn
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name mitbih --model_name attention_lstm_rnn
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name mitbih --model_name vanilla_rnn --transfer_learning --rnn_freeze permanent
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name mitbih --model_name lstm_rnn --transfer_learning --rnn_freeze permanent
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name mitbih --model_name vanilla_rnn --rnn_bidirectional --transfer_learning --rnn_freeze permanent
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name mitbih --model_name lstm_rnn --rnn_bidirectional --transfer_learning --rnn_freeze permanent
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name mitbih --model_name vanilla_rnn --transfer_learning --rnn_freeze temporary
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name mitbih --model_name lstm_rnn --transfer_learning --rnn_freeze temporary
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name mitbih --model_name vanilla_rnn --rnn_bidirectional --transfer_learning --rnn_freeze temporary
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name mitbih --model_name lstm_rnn --rnn_bidirectional --transfer_learning --rnn_freeze temporary
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name mitbih --model_name vanilla_rnn --transfer_learning --rnn_freeze never
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name mitbih --model_name lstm_rnn --transfer_learning --rnn_freeze never
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name mitbih --model_name vanilla_rnn --rnn_bidirectional --transfer_learning --rnn_freeze never
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name mitbih --model_name lstm_rnn --rnn_bidirectional --transfer_learning --rnn_freeze never

bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name ptbdb --model_name vanilla_rnn
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name ptbdb --model_name lstm_rnn
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name ptbdb --model_name vanilla_rnn --rnn_bidirectional
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name ptbdb --model_name lstm_rnn --rnn_bidirectional
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name ptbdb --model_name vanilla_cnn
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name ptbdb --model_name residual_cnn
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name ptbdb --model_name ae
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name ptbdb --model_name inception_net
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name ptbdb --model_name attention_vanilla_rnn
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name ptbdb --model_name attention_lstm_rnn
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name ptbdb --model_name vanilla_rnn --transfer_learning --rnn_freeze permanent
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name ptbdb --model_name lstm_rnn --transfer_learning --rnn_freeze permanent
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name ptbdb --model_name vanilla_rnn --rnn_bidirectional --transfer_learning --rnn_freeze permanent
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name ptbdb --model_name lstm_rnn --rnn_bidirectional --transfer_learning --rnn_freeze permanent
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name ptbdb --model_name vanilla_rnn --transfer_learning --rnn_freeze temporary
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name ptbdb --model_name lstm_rnn --transfer_learning --rnn_freeze temporary
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name ptbdb --model_name vanilla_rnn --rnn_bidirectional --transfer_learning --rnn_freeze temporary
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name ptbdb --model_name lstm_rnn --rnn_bidirectional --transfer_learning --rnn_freeze temporary
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name ptbdb --model_name vanilla_rnn --transfer_learning --rnn_freeze never
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name ptbdb --model_name lstm_rnn --transfer_learning --rnn_freeze never
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name ptbdb --model_name vanilla_rnn --rnn_bidirectional --transfer_learning --rnn_freeze never
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py --dataset_name ptbdb --model_name lstm_rnn --rnn_bidirectional --transfer_learning --rnn_freeze never

bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python ensemble.py --dataset_name mitbih --model_name majority_ensemble
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python ensemble.py --dataset_name mitbih --model_name log_reg_ensemble

bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python ensemble.py --dataset_name ptbdb --model_name majority_ensemble
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python ensemble.py --dataset_name ptbdb --model_name log_reg_ensemble
