export SLURM_JOB_ID=3821067
srun \
 -p MoE \
 -w SH-IDCA1404-10-140-54-37\
 deepspeed --num_gpus 1 --master_port=29508 env/scienceworld/generate_subtask.py