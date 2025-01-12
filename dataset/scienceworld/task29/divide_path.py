import json
import re

task_id = 29
for vari in range(270):
    with open(f'dataset/scienceworld/task{task_id}/variation{vari}.json', 'r') as json_file:
        raw_data = json.load(json_file)
    raw_actions = raw_data['action']
    flag = 0
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
            if flag>=2:
                flag = 0
                # group_actions.append(action_segment)
                # action_segment = []
                group_actions.append(action_segment)
                action_segment = []
                action_segment.append(action)
            else:
                flag+=1
                continue
        else:
            action_segment.append(action)
    group_actions.append(action_segment)
    # print("############")
    # print(group_actions)
    subtask1= [
        "Navigation to kitchen",
        "Find the thermometer and focus it",
        "Navigation to {loc1}",
        "Find the {obj} and focus it",
        "Navigation to {loc2}",
        "Measure if the temperature of {obj} is above {temp} degree",
        "Move {obj} to the {color} box"
              ]
    subtask2 = [
        "Navigation to kitchen",
        "Find the thermometer and focus it",
        "Navigation to {loc1}",
        "Find the {obj} and focus it",
        "Measure if the temperature of {obj} is above {temp} degree",
        "Move {obj} to the {color} box"
    ]
    if len(group_actions)==7:
        pass
    elif len(group_actions)==6:
        pass
    else:
        print("*************error")
        exit()

    subtask = subtask1 if len(group_actions)==7 else subtask2
    # print(re.search(r"is located around the (\w+).", raw_data['task_description']))
    loc1 = re.search(r"is located around the (.*?)\.", raw_data['task_description']).group(1)
    obj = re.search(r"task is to measure the temperature of (.*?)\,", raw_data['task_description']).group(1)
    loc2 = re.search(r"The boxes are located around the (.*?)\.", raw_data['task_description']).group(1)
    temp = re.search(r"temperature is above (.*?)\ degrees", raw_data['task_description']).group(1)
    color = re.search(r"place it in the (.*?)\ box.", raw_data['task_description']).group(1)
    for i in range(len(subtask)):
        subtask[i] = subtask[i].replace("{obj}",obj)
        subtask[i] = subtask[i].replace("{loc1}", loc1)
        subtask[i] = subtask[i].replace("{loc2}", loc2)
        subtask[i] = subtask[i].replace("{temp}", temp)
        subtask[i] = subtask[i].replace("{color}", color)

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
    
        

