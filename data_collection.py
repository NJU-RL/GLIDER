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


with open("./config/collection.json", 'r') as f:
    args = json.load(f)
print(args)


from alg.eval_policy import EvalAgent

data_container={
    'task_description': [],
    'obs': [],
    'action': [],
    'next_obs': [],
    'reward': [],
    'score': [],
    'done': []
    }

eval_agent = EvalAgent(args)
vari_nums = pd.read_csv(f"env/{args['benchmark']}/task_nums.csv",encoding='utf-8')['train'].tolist()
for task_id, vari_num in enumerate(vari_nums):
    for vari_id in list(range(vari_num)):
        eval_agent.data_collect(task_id, vari_id, data_container, [0,100])
with open('score_0_100.json', 'w') as json_file:
    json.dump(data_container, json_file, indent=4)
