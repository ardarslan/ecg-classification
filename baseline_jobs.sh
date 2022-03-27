cd src/baselines

bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python baseline_mitbih.py
bsub -n 2 -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python baseline_ptbdb.py
