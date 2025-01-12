import json
import re

def group_actions(raw_actions, task_description):
    grouped_actions = []
    subtasks = []
    i = 0
    
    # 获取植物类型和种子位置
    plant_type = re.search(r"grow a (\w+)", task_description).group(1)
    seed_location = re.search(r"Seeds can be found in the (.*?)\.", task_description).group(1)
    
    # 1. 获取种子jar阶段
    seed_jar_idx = next(i for i, action in enumerate(raw_actions) if "pick up seed jar" in action)
    grouped_actions.append(raw_actions[:seed_jar_idx + 1])
    subtasks.append(f"Navigate to {seed_location} then get seed jar")
    i = seed_jar_idx + 1
    
    # 2. 导航到温室阶段
    greenhouse_idx = next(i for i, action in enumerate(raw_actions[i:], i) if "look around" in action)
    grouped_actions.append(raw_actions[i:greenhouse_idx + 1])
    subtasks.append("Navigate to greenhouse")
    i = greenhouse_idx + 1
    
    # 3. 准备土壤阶段（如果存在）
    soil_actions = []
    while i < len(raw_actions):
        if f"move {plant_type} seed" in raw_actions[i]:
            if soil_actions:
                grouped_actions.append(soil_actions)
                subtasks.append("Prepare soil for planting")
            break
        soil_actions.append(raw_actions[i])
        i += 1
    
    # 4. 播种和观察阶段
    seed_actions = []
    while i < len(raw_actions):
        if "activate sink" in raw_actions[i]:
            grouped_actions.append(seed_actions)
            subtasks.append(f"Plant and observe {plant_type} seed")
            break
        seed_actions.append(raw_actions[i])
        i += 1
    
    # 5. 循环处理浇水和等待阶段
    current_actions = []
    while i < len(raw_actions):
        if "activate sink" in raw_actions[i]:
            # 收集浇水动作
            current_actions = []
            while i < len(raw_actions) and "deactivate sink" not in raw_actions[i]:
                current_actions.append(raw_actions[i])
                i += 1
            if i < len(raw_actions):
                current_actions.append(raw_actions[i])  # 添加 deactivate sink
                i += 1
                grouped_actions.append(current_actions)
                subtasks.append("Water the plant")
                
        elif raw_actions[i] == "wait1":
            # 收集等待动作
            current_actions = []
            while i < len(raw_actions) and (raw_actions[i] == "wait1" or raw_actions[i] == "look around"):
                current_actions.append(raw_actions[i])
                i += 1
            if current_actions:
                grouped_actions.append(current_actions)
                subtasks.append("Wait for growth")
                
        else:
            i += 1
    
    return grouped_actions, subtasks

task_id = 11  # 或其他任务ID
for vari in range(62):
    with open(f'dataset/scienceworld/task{task_id}/variation{vari}.json', 'r') as json_file:
        raw_data = json.load(json_file)
    raw_actions = raw_data['action']
    task_description = raw_data['task_description']
    
    grouped_actions, subtasks = group_actions(raw_actions, task_description)
    
    # 验证action完整性
    flat_actions = [action for group in grouped_actions for action in group]
    assert len(flat_actions) == len(raw_actions), "Action count mismatch"
    assert flat_actions == raw_actions, "Action sequence mismatch"

    result = {
        "grouped_actions": grouped_actions,
        "subtasks": subtasks
    }
    raw_data['group_action'] = grouped_actions
    raw_data['subtask'] = subtasks
    if vari==1:
        # print(raw_data)
        print(raw_data['group_action'])
        # print(raw_data)
    with open(f'dataset/scienceworld/task{task_id}/variation{vari}.json', 'w') as json_file:
        json.dump(raw_data, json_file, indent=4)
    
    
        

