export SLURM_JOB_ID=4021537
srun \
 -p MoE \
 -w SH-IDCA1404-10-140-54-102\
 deepspeed --num_gpus 1 --master_port=29520 eval_glider.py