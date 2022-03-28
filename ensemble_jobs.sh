cd src

# Ensemble: Majority Voting for MIT-BIH dataset
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python ensemble.py --dataset_name mitbih --model_name majority_ensemble

# Ensemble: Logistic Regression for MIT-BIH dataset
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python ensemble.py --dataset_name mitbih --model_name log_reg_ensemble

# Ensemble: Majority Voting for PTBDB dataset
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python ensemble.py --dataset_name ptbdb --model_name majority_ensemble

# Ensemble: Logistic Regression for PTBDB dataset
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python ensemble.py --dataset_name ptbdb --model_name log_reg_ensemble

