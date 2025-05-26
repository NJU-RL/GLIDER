export SLURM_JOB_ID=3925846
srun \
 -p MoE \
 -w SH-IDCA1404-10-140-54-102\
 deepspeed --num_gpus 1 --master_port=29508 eval.py