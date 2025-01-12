from scienceworld import ScienceWorldEnv
import json
import os


TASK_NUM = 30

env = ScienceWorldEnv("", envStepLimit=1000)
task_names = env.getTaskNames()
data_type = "train"

for task_id in range(TASK_NUM):
    env.load(task_names[task_id], generateGoldPath=True)
    if data_type == "train":
        vari_ids = env.getVariationsTrain()
    
    path = f"dataset/scienceworld/task{task_id}/"
    'variation{vari}.json'
    os.makedirs(path, exist_ok=True)
    for vari in vari_ids:
        env.load(task_names[task_id], vari, generateGoldPath=True)
        task_description = env.taskdescription()
        with open(f'dataset/scienceworld/task{task_id}/variation{vari}.json', 'r') as json_file:
            gold_data = json.load(json_file)
        # gold_data["obs"] = []
        # gold_data["next_obs"] = []
        # gold_data["reward"] = []
        # gold_data["score"] = []
        # gold_data["done"] = []
        if isinstance(gold_data["action"][0],str):
            action_traj = gold_data["action"]
            # gold_data["group_action"] = []
        else:
            action_traj = []
            for traj in gold_data['action']:
                for a in traj:
                    action_traj.append(a)
            gold_data["group_action"] = gold_data["action"]
            gold_data["action"] = action_traj

        # obs, _ = env.reset()
        # for action in action_traj:
        #     obs_, reward, isCompleted, infos = env.step(action)
            
        #     gold_data["obs"].append(obs)
        #     gold_data["reward"].append(reward/100)
        #     gold_data["score"].append(infos["score"]/100)
        #     gold_data["done"].append(isCompleted)
        #     gold_data["next_obs"].append(obs_)
        #     obs = obs_
        # print(gold_data)
        
        with open(path+f'variation{vari}.json', 'w') as json_file:
            json.dump(gold_data, json_file, indent=4)
            print(task_id, vari,"   done")
        # exit()

