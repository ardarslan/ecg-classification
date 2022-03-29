# Environment Setup

conda env create -f environment.yml

conda activate ml4hc_project1

# Reproduce results as a batch

## Baselines

chmod +x baseline_jobs.sh

./baseline_jobs.sh

## Our models

chmod +x main_jobs.sh

./main_jobs.sh

## Ensemble models

chmod +x ensemble_jobs.sh

./ensemble_jobs.sh

# Reproduce one experiment at a time
Please see .sh files.


Results will be saved in checkpoints folder.