import os
import GPUtil
import json

def get_free_gpus(num_gpus_needed):
    # 使用GPUtil获取空闲GPU
    available_gpus = GPUtil.getAvailable(order='memory', limit=num_gpus_needed, maxLoad=0.1, maxMemory=0.1, includeNan=False, excludeID=[], excludeUUID=[])
    return available_gpus

num_gpus = int(os.environ.get('WORLD_SIZE', 1))

free_gpus = get_free_gpus(num_gpus)
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, free_gpus))


with open("./config/default.json", 'r') as f:
    args = json.load(f)
print(args)


from alg.eval_policy import EvalAgent

eval_agent = EvalAgent(args)

eval_agent.eval_policy('test')
