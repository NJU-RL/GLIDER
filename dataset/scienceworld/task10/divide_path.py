import json
import re

def find_sequence_end(actions, start_idx, end_markers):
    """找到序列的结束位置"""
    i = start_idx
    while i < len(actions):
        if any(marker in actions[i] for marker in end_markers):
            return i
        i += 1
    return i

def group_actions(raw_actions, plant_type,seed_location):
    grouped_actions = []
    subtasks = []
    i = 0
    
    # 1. 找种子jar阶段
    seed_jar_idx = next(i for i, action in enumerate(raw_actions) if "pick up seed jar" in action)
    grouped_actions.append(raw_actions[:seed_jar_idx + 1])
    subtasks.append(f"Navigete to {seed_location} then get seed jar")
    i = seed_jar_idx + 1
    
    # 2. 导航到温室阶段
    greenhouse_idx = next(i for i, action in enumerate(raw_actions[i:], i) if "look around" in action)
    grouped_actions.append(raw_actions[i:greenhouse_idx + 1])
    subtasks.append("Navigate to greenhouse")
    i = greenhouse_idx + 1
    
    # 3. 准备土壤阶段
    seed_start_idx = next(i for i, action in enumerate(raw_actions[i:], i) 
                         if f"move {plant_type} seed" in action)
    grouped_actions.append(raw_actions[i:seed_start_idx])
    subtasks.append("Prepare soil for flower pots")
    i = seed_start_idx
    
    # 4. 播种阶段
    sink_start_idx = next(i for i, action in enumerate(raw_actions[i:], i) 
                         if "activate sink" in action)
    grouped_actions.append(raw_actions[i:sink_start_idx])
    subtasks.append(f"Plant {plant_type} seeds")
    i = sink_start_idx
    
    # 5. 处理剩余的循环阶段（浇水、等待、环境控制等）
    while i < len(raw_actions):
        if "activate sink" in raw_actions[i]:
            # 浇水阶段
            end_idx = next(j for j, action in enumerate(raw_actions[i:], i) 
                          if "deactivate sink" in action) + 1
            grouped_actions.append(raw_actions[i:end_idx])
            subtasks.append("Water the plants")
            i = end_idx
            
        elif raw_actions[i] == "wait1":
            # 等待阶段
            start_idx = i
            while i < len(raw_actions) and (raw_actions[i] == "wait1" or raw_actions[i] == "look around"):
                i += 1
            grouped_actions.append(raw_actions[start_idx:i])
            subtasks.append("Wait for growth")
            
        elif "close door" in raw_actions[i] or "open bee hive" in raw_actions[i]:
            # 环境控制阶段
            start_idx = i
            while i < len(raw_actions) and ("door" in raw_actions[i] or "bee hive" in raw_actions[i]):
                i += 1
            grouped_actions.append(raw_actions[start_idx:i])
            subtasks.append("Release bees for pollination")
            
        elif "focus" in raw_actions[i]:
            # 最终观察阶段
            grouped_actions.append(raw_actions[i:])
            subtasks.append(f"focus on the grown {plant_type}")
            break
            
        else:
            i += 1

    return grouped_actions, subtasks

task_id = 10
for vari in range(62):
    with open(f'dataset/scienceworld/task{task_id}/variation{vari}.json', 'r') as json_file:
        raw_data = json.load(json_file)
    raw_actions = raw_data['action']

    plant_type = re.search(r"grow a (\w+)", raw_data['task_description']).group(1)
    seed_location = re.search(r"Seeds can be found in the (.*?)\.", raw_data['task_description']).group(1)
    grouped_actions, subtasks = group_actions(raw_actions, plant_type, seed_location)
    
    # 验证action完整性
    print(vari)
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
        print(raw_data)
    with open(f'dataset/scienceworld/task{task_id}/variation{vari}.json', 'w') as json_file:
        json.dump(raw_data, json_file, indent=4)
    
    
        

