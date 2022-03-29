cd src

# Ensemble: Majority Voting for MIT-BIH dataset
python ensemble.py --dataset_name mitbih --model_name majority_ensemble

# Ensemble: Logistic Regression for MIT-BIH dataset
python ensemble.py --dataset_name mitbih --model_name log_reg_ensemble

# Ensemble: Majority Voting for PTBDB dataset
python ensemble.py --dataset_name ptbdb --model_name majority_ensemble

# Ensemble: Logistic Regression for PTBDB dataset
python ensemble.py --dataset_name ptbdb --model_name log_reg_ensemble

