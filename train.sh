# export SLURM_JOB_ID=3904553
srun \
 -c 90 \
 -p MoE \
 -w SH-IDCA1404-10-140-54-89\
 deepspeed --num_gpus 8 --master_port=29514 train.py