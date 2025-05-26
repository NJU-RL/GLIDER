from scienceworld import ScienceWorldEnv
import json
import os
import copy
from prompt.inst import action_gene_prompt

# with open("./env/action_space.json", 'r') as f:
#     actions_list = json.load(f)
# action_gene_prompt = action_gene_prompt.replace("[getPossibleActions]",str(actions_list))

TASK_NUM = 30

env = ScienceWorldEnv("", envStepLimit=1000)
task_names = env.getTaskNames()
data_type = "train"

for task_id in range(TASK_NUM):
    env.load(task_names[task_id], generateGoldPath=True)
    if data_type == "train":
        vari_ids = env.getVariationsTrain()
    elif data_type == "dev":
        vari_ids = env.getVariationsDev()
    elif data_type == "test":
        vari_ids = env.getVariationsTest()
    path = f"dataset/scienceworld/task{task_id}/"
    os.makedirs(path, exist_ok=True)
    for vari in vari_ids:
        env.load(task_names[task_id], vari, generateGoldPath=True)
        task_description = env.taskdescription()
        gold_data = {
            "task_description":task_description,
            "subtask":[],
            "action":[]
            # "next_obs":[],
            # "reward":[],
            # "score":[],
            # "done":[],
        }

        obs, _ = env.reset()
        # obj = env.getPossibleObjects()
        gold_data["action"] = env.getGoldActionSequence()
        # for action in gold_data["action"]:
        #     obs_, reward, isCompleted, infos = env.step(action)
            
            # gold_data["obs"].append(obs)
            # gold_data["reward"].append(reward/100)
            # gold_data["score"].append(infos["score"]/100)
            # gold_data["done"].append(isCompleted)

            # history_actions.append(action)
            # gold_data["next_obs"].append(obs_)
            # print(history_actions)
            # obs = obs_
        
        with open(path+f'variation{vari}.json', 'w') as json_file:
            json.dump(gold_data, json_file, indent=4)
            print(task_id, vari,"   done")
            