cd src

bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python ensemble.py --dataset_name mitbih --model_name majority_ensemble
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python ensemble.py --dataset_name mitbih --model_name log_reg_ensemble

bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python ensemble.py --dataset_name ptbdb --model_name majority_ensemble
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python ensemble.py --dataset_name ptbdb --model_name log_reg_ensemble
