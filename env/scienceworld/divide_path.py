from scienceworld import ScienceWorldEnv
import json
import os
import re
import numpy as np



def extract_subtasks_and_actions(raw_data):
    subtask_list = []
    action_list = []

    # 分割子任务（以数字开头的部分）
    lines = re.split(r"(?=\d+\.\s\*\*|\d+\.\s)", raw_data.strip())

    for line in lines:
        # 使用正则表达式来匹配子任务和动作部分
        parts = re.split(r":\s*\n\s*- Actions:\s*|\s*:\s*\{|\s*:\s*", line, 1)

        if len(parts) >= 2:
            # 处理子任务部分，去除**符号并替换下划线为空格
            subtask = re.sub(r"\*\*|\n", "", parts[0].strip()).replace('_', ' ')
            # 去掉子任务名称前的序号和多余的空格
            subtask = re.sub(r"^\d+\.\s*", "", subtask).strip()
            subtask_list.append(subtask)

            # 处理动作部分，兼容多种符号，如花括号、反引号等
            actions_block = parts[1].strip()
            # 去掉花括号和反引号
            actions_block = re.sub(r"[{}']", "", actions_block)
            actions_block = re.sub(r"`", "", actions_block)
            # 处理换行和破折号，并转换成逗号分隔的列表
            actions_cleaned = [action.replace("-","").strip() for action in re.split(r",\s*|\n\s*- ", actions_block) if action]
            action_list.append(actions_cleaned)
        # else:
        #     print(f"Warning: No actions found for the line: {line}")

    return subtask_list, action_list


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
    save_path = f"datasets/hrl_data/task{task_id}/{data_type}/"
    os.makedirs(save_path, exist_ok=True)
    for vari in vari_ids:
        with open(f'datasets/subtask_data/task{20}/{data_type}/variation{168}.json', 'r') as json_file:
            raw_data = json.load(json_file)

        subtasks, actions = extract_subtasks_and_actions(raw_data['subtask'])
        # print(subtasks)
        # print(actions)
        # exit()
        # subtasks_with_colon = re.findall(r'\d+\.\s\*\*(.*?)\*\*', raw_data['subtask'])        
        # actions = [re.findall(r'`([^`]+)`', action_block) for action_block in re.split(r'\d+\.\s\*\*.*?\*\*', raw_data['subtask'])[1:]]
        # subtasks = [subtask.replace(":","") for subtask in subtasks_with_colon]
        # if not int(np.sum([len(a) for a in actions])):
        #     actions = [re.findall(r'\{(.*?)\}', action_block) for action_block in re.split(r'\d+\.\s\*\*.*?\*\*:', raw_data['subtask'])[1:]]

        raw_data['subtask_action'] = raw_data['subtask']
        raw_data['subtask'] = subtasks
        raw_data['action_list'] = actions
        error = np.sum([len(a) for a in actions])!=len(raw_data['action'])
        if not error:
            with open(save_path+f'variation{vari}.json', 'w') as json_file:
                json.dump(raw_data, json_file, indent=4)
        else:
            print("error:", len(actions),task_id,vari, "pro", np.sum([len(a) for a in actions]), len(raw_data['action']))