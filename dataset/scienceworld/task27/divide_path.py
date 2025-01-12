import json
import re

task_id = 27
for vari in range(450):
    with open(f'dataset/scienceworld/task{task_id}/variation{vari}.json', 'r') as json_file:
        raw_data = json.load(json_file)
    raw_actions = raw_data['action']
    flag = False
    navi_flag = False
    group_actions = []
    action_segment = []
    for action in raw_actions:
        if action.startswith('wait1'):
            continue
        elif action.startswith('drop'):
            # action_segment = []
            # group_actions.append(action_segment)
            # action_segment = []
            # action_segment.append(action)
            continue 
        elif action.startswith('pick up') or action.startswith('move'):
            if action.startswith('pick up'):
                navi_flag = True
            group_actions.append(action_segment)
            action_segment = []
            action_segment.append(action)
        elif action.startswith('open door to'):
            if navi_flag == True:
                navi_flag = False
                group_actions.append(action_segment)
                action_segment = []
                action_segment.append(action)
            else:
                action_segment.append(action)
        elif action.startswith('look around'):
            if flag==False:
                flag = True
                # group_actions.append(action_segment)
                # action_segment = []
                group_actions.append(action_segment)
                action_segment = []
                action_segment.append(action)
            else:
                continue
        else:
            action_segment.append(action)
    group_actions.append(action_segment)
    # print("############")
    # print(group_actions)
    subtask1= [
        "Navigation to {loc}",
        "Find the {obj} and focus it",
        "Determine if {obj} is electrically conductive",
        "Move {obj} to the {color} box"
              ]
    subtask2 = [
        "Navigation to {loc}",
        "Find the {obj} and focus it",
        "Navigation to workshop",
        "Determine if {obj} is electrically conductive",
        "Move {obj} to the {color} box"
    ]
    subtask = subtask1 if navi_flag else subtask2
    # print(re.search(r"is located around the (\w+).", raw_data['task_description']))
    location = re.search(r"is located around the (.*?)\.", raw_data['task_description']).group(1)

    print(location)
    subtask[0] = subtask[0].replace("{loc}",location)
    
    obj = re.search(r"task is to determine if (.*?) is electrically", raw_data['task_description']).group(1)
    print(obj)
    for i in range(len(subtask)):
        subtask[i] = subtask[i].replace("{obj}",obj)

    print(raw_data['action'][-2])
    if raw_data['action'][-2] == '0':
        ex_action = raw_data['action'][-3]
    else:
        ex_action = raw_data['action'][-2]
    color = re.search(r"to (.*?) box", ex_action).group(1)
    subtask[-1] = subtask[-1].replace("{color}",color)

    raw_data['action'] = group_actions
    raw_data['subtask'] = subtask
    print("###")
    print(vari)
    print(raw_data)
    print(navi_flag)
    # exit()

    
    with open(f'dataset/scienceworld/task{task_id}/variation{vari}.json', 'w') as json_file:
        json.dump(raw_data, json_file, indent=4)
    
        

