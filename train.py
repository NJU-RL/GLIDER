import os
import GPUtil
import json

def get_free_gpus(num_gpus_needed):
    # Use GPUtil to get free GPU
    available_gpus = GPUtil.getAvailable(order='memory', limit=num_gpus_needed, maxLoad=0.1, maxMemory=0.1, includeNan=False, excludeID=[], excludeUUID=[])
    return available_gpus

num_gpus = int(os.environ.get('WORLD_SIZE', 1))

free_gpus = get_free_gpus(num_gpus)
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, free_gpus))

DEBUG_MODE = False
if DEBUG_MODE:
    rank = int(os.environ.get("RANK", 0))
    import debugpy

    debugpy.listen(address = ('0.0.0.0', 5678 + rank))
    if rank == 0:
        debugpy.wait_for_client() 
    breakpoint()

with open("./config/default.json", 'r') as f:
    args = json.load(f)
print(args)

if args['alg_name']=="awac":
    from alg.awac import Agent
elif args['alg_name']=="bc":
    from alg.bc import Agent




agent = Agent(args)
agent.learn()