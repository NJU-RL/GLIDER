import os
import GPUtil
import json
import pandas as pd
def get_free_gpus(num_gpus_needed):
    # 使用GPUtil获取空闲GPU
    available_gpus = GPUtil.getAvailable(order='memory', limit=num_gpus_needed, maxLoad=0.1, maxMemory=0.1, includeNan=False, excludeID=[], excludeUUID=[])
    return available_gpus

num_gpus = int(os.environ.get('WORLD_SIZE', 1))

free_gpus = get_free_gpus(num_gpus)
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, free_gpus))


with open("./config/collection_glider.json", 'r') as f:
    args = json.load(f)
print(args)

from alg.eval_glider import EvalAgent

eval_agent = EvalAgent(args)
eval_agent.load_policy(args['collect_model_path'])

high_data_container = {
    'task_description': [],
    'obs': [],
    'subtask': [],
    'reward': [],
    'score': [],
    'done': []
    }
low_data_container = {
    'subtask': [],
    'obs': [],
    'action': [],
    'reward': [],
    'score': [],
    'done': []
}

import random
import numpy as np
import torch

random.seed(args['seed'])
np.random.seed(args['seed'])
torch.manual_seed(args['seed'])

vari_nums = pd.read_csv(f"env/{args['benchmark']}/task_nums.csv",encoding='utf-8')['train'].tolist()
for task_id, vari_num in enumerate(vari_nums):
    if vari_num == 0:
        continue
    for _ in range(args['collect_vari_num']):
        if args['half'] == 0:
            vari_id = random.choice(range(vari_num//2, vari_num))
        elif args['half'] == 1:
            vari_id = random.choice(range(0, vari_num//2))
        eval_agent.data_collect(task_id, vari_id, high_data_container, low_data_container)
        with open(f"dataset/{args['benchmark']}/high_data/medium{args['half']}_seed{args['seed']}.json", 'w') as json_file:
            json.dump(high_data_container, json_file, indent=4)
        with open(f"dataset/{args['benchmark']}/low_data/medium{args['half']}_seed{args['seed']}.json", 'w') as json_file:
            json.dump(low_data_container,json_file, indent=4)
