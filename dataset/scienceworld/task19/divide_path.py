import json
import re

task_id = 19
for vari in range(62):
    with open(f'dataset/scienceworld/task{task_id}/variation{vari}.json', 'r') as json_file:
        raw_data = json.load(json_file)
    raw_actions = raw_data['action']
    
    group_actions = []
    action_segment = []
    for action in raw_actions:
        if action.startswith('focus on'):
            group_actions.append(action_segment)
            action_segment = []
            action_segment.append(action)
        else:
            action_segment.append(action)
    group_actions.append(action_segment)
    # print("############")
    # print(group_actions)
    raw_data['action'] = group_actions

    subtask= ["Navigation to {kitchen}",
              "Find the shortest life span animal and focus it"]

    location = re.search(r"in the '(\w+)' location", raw_data['task_description']).group(1)
    
    subtask[0] = subtask[0].replace("{kitchen}",location)
    
    raw_data['subtask'] = subtask
    print("###")
    print(vari)
    print(raw_data)
    # exit()

    
    with open(f'dataset/scienceworld/task{task_id}/variation{vari}.json', 'w') as json_file:
        json.dump(raw_data, json_file, indent=4)
    
        

