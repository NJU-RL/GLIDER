export SLURM_JOB_ID=4021551
srun \
 -p MoE \
 -w SH-IDCA1404-10-140-54-37\
 deepspeed --num_gpus 4 --master_port=29518 train_glider_online.py