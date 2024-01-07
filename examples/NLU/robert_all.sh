#!/bin/bash

# List of bash scripts to run
scripts=("roberta_base_cola.sh" "roberta_base_mnli.sh" "roberta_base_mrpc.sh" "roberta_base_qnli.sh" "roberta_base_qqp.sh" "roberta_base_rte.sh" "roberta_base_sst2.sh" "roberta_base_stsb.sh")

# Run each script
for script in "${scripts[@]}"; do
    echo "Running $script"
    bash "$script"
    echo "Completed $script"
done