import json
import re

for vari in range(150):
    with open(f'dataset/scienceworld/task{5}/variation{vari}.json', 'r') as json_file:
        raw_data = json.load(json_file)
    raw_actions = raw_data['action']
    
    group_actions = []
    action_segment = []
    for action in raw_actions:
        if action.startswith('pick up'):
            group_actions.append(action_segment)
            action_segment = []
            action_segment.append(action)
        elif action.startswith('move'):
            group_actions.append(action_segment)
            action_segment = []
            action_segment.append(action)
        else:
            action_segment.append(action)
    group_actions.append(action_segment)
    # print("############")
    # print(group_actions)
    raw_data['action'] = group_actions

    subtask= ["Find an animal and focus it",
              "Navigation to {kitchen} with the animal",
              "Move the animal to the {red} box in the {kitchen}"]
    
    if raw_actions[-2].startswith("go to"):
        location = raw_actions[-2].partition("to ")[-1]
        subtask[1] = subtask[1].replace("{kitchen}",location)
        subtask[-1] = subtask[-1].replace("{kitchen}",location)
    else:
        print("error")
    color = re.search(r'to (\w+) box', raw_actions[-1]).group(1)
    subtask[-1]= subtask[-1].replace("{red}",color)
    raw_data['subtask'] = subtask
    print("###")
    print(subtask)
    exit()

    
    with open(f'dataset/scienceworld/task{5}/variation{vari}.json', 'w') as json_file:
        json.dump(raw_data, json_file, indent=4)
    
        

