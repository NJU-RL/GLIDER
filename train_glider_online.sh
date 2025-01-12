export SLURM_JOB_ID=3925555
srun \
 -p MoE \
 -w SH-IDCA1404-10-140-54-35\
 deepspeed --num_gpus 1 --master_port=29506 train_glider_online.py