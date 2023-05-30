# Activate venv
source iso_env/bin/activate 

# Set MASTER_PORT and MASTER_ADDRESS 
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

module load python/3.7.4

# This is an example script where fine-tune BERT on SST2 with I-STAR regularization. 
# When performing I-STAR regularization, we use a shrinkage parameter (zeta) of 0.2 and a tuning parameter (lambda) of -5 to decrease isotropy in the embedding space. 
# We apply the I-STAR penalty from the union of all hidden states. You can apply I-STAR to individual layers by altering --layer

for SEED in 0; do 
# Run script. See run_model.py file for options.  
CUDA_LAUNCH_BLOCKING=1 python3 run_model.py --task "sst2" --model "bert" --regularizer "istar" --training "True" --zeta 0.2 --tuning_param -5 --seed $SEED --num_epochs 4 --layer "all" --batch_size 32; 
done;
