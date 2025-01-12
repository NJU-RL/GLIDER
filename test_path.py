
# with open("./env/action_space.json", 'r') as f:
#     actions_list = json.load(f)
# action_gene_prompt = action_gene_prompt.replace("[getPossibleActions]",str(actions_list))

from scienceworld import ScienceWorldEnv

TASK_NUM = 30
import json

env = ScienceWorldEnv("", envStepLimit=1000)
task_names = env.getTaskNames()
taskname2id = {}
for i, task_name in enumerate(task_names):
     taskname2id[task_name] = i
with open(f'env/scienceworld/taskname2id.json', 'w') as json_file:
        json.dump(taskname2id, json_file, indent=4)

data_type = "train"

task_id, vari_id = 10, 0
env.load(task_names[task_id], vari_id, generateGoldPath=False)
task_description = env.taskdescription()
subtask = [
    "Navigate to kitchen",
        "Prepare temperature and metal pot",
        "Fill metal pot with water",
        "Focus on substance",
        "Freeze water",
        "Monitor water temperature until it freezes"
]
action_traj = [
        "open door to kitchen",
        "go to kitchen",
        "look around",
        "pick up seed jar",
        "open door to outside",
        "go to outside",
        "open door to greenhouse",
        "go to greenhouse",
        "look around",
        "move apple seed in seed jar to flower pot 3",
        "0",
        "move apple seed in seed jar to flower pot 2",
        "0",
        "move apple seed in seed jar to flower pot 1",
        "0",
        "activate sink",
        "move jug to sink",
        "pour jug into flower pot 3",
        "move jug to sink",
        "pour jug into flower pot 2",
        "move jug to sink",
        "pour jug into flower pot 1",
        "deactivate sink",
        "wait1",
        "wait1",
        "wait1",
        "wait1",
        "wait1",
        "look around",
        "activate sink",
        "move jug to sink",
        "pour jug into flower pot 3",
        "move jug to sink",
        "pour jug into flower pot 2",
        "move jug to sink",
        "pour jug into flower pot 1",
        "deactivate sink",
        "wait1",
        "wait1",
        "wait1",
        "wait1",
        "wait1",
        "look around",
        "activate sink",
        "move jug to sink",
        "pour jug into flower pot 3",
        "move jug to sink",
        "pour jug into flower pot 2",
        "move jug to sink",
        "pour jug into flower pot 1",
        "deactivate sink",
        # "close door to hallway",
        # "close door to outside",
        "open bee hive",
        "wait1",
        "wait1",
        "wait1",
        "wait1",
        "wait1",
        "wait1",
        "wait1",
        "wait1",
        "wait1",
        "look around",
        "activate sink",
        "move jug to sink",
        "pour jug into flower pot 3",
        "move jug to sink",
        "pour jug into flower pot 2",
        "move jug to sink",
        "pour jug into flower pot 1",
        "deactivate sink",
        "focus on apple in apple tree",
        "0",
        "wait1",
        "wait1",
        "wait1",
        "wait1",
        "wait1",
        "look around",
        "wait1"
    ]

    
actions = action_traj
# actions = []
# for traj in action_traj:
#     for a in traj:
#         actions.append(a)
print(task_description)
print("-----------------------")
obs, _ = env.reset()
for action in actions:
    obs_, reward, isCompleted, infos = env.step(action)
    print(f"reward:{infos['score']}, done:{isCompleted}, action:{action}")
    print(obs_)
    if obs_ == "No known action matches that input.":
        exit()

        # obj = env.getPossibleObjects()
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
        
            