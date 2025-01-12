import json
import re
task_id = 7
for vari in range(150):
    if vari in [73]:
        continue
    with open(f'dataset/scienceworld/task{task_id}/variation{vari}.json', 'r') as json_file:
        raw_data = json.load(json_file)

    # print(raw_data)
    raw_actions = raw_data['action']
    
    group_actions = []
    action_segment = []
    for action in raw_actions:
            
        if action.startswith('focus on'):
            action_segment.append(action)
            group_actions.append(action_segment)
            action_segment = []
        else:
            action_segment.append(action)
        # print(action_segment)

    group_actions.append(action_segment)
    # print("############")
    # print(group_actions)
    raw_data['action'] = group_actions

    subtask= ["Find non-living thing in {kitchen} and focus it",
              "Move the non-living thing to the {red} box in the {kitchen}"]
    location = raw_data['task_description'].partition("in the")[-1].replace(".","")
    subtask[0] = subtask[0].replace("{kitchen}",location)
    subtask[1] = subtask[1].replace("{kitchen}",location)
    print("*****")
    print(vari)
    color = re.search(r'to (\w+) box', raw_actions[-1]).group(1)
    subtask[1]= subtask[1].replace("{red}",color)
    raw_data['subtask'] = subtask
    # print("###")
    # print(subtask)
    print(raw_data)

    
    with open(f'dataset/scienceworld/task{task_id}/variation{vari}.json', 'w') as json_file:
        json.dump(raw_data, json_file, indent=4)
    
        

