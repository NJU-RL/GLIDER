import json

task_id = 29
for vari in range(450):
    with open(f'dataset/scienceworld/task{task_id}/variation{vari}.json', 'r') as json_file:
        raw_data = json.load(json_file)
    actions = raw_data['action']
    subtask = []
    action_group = []
    for i in range(len(raw_data['subtask'])):
        if len(raw_data['action'][i]) != 0:
            subtask.append(raw_data['subtask'][i])
            action_group.append(raw_data['action'][i])

    raw_data["action"] = action_group
    raw_data['subtask'] = subtask

    print(raw_data)
    with open(f'dataset/scienceworld/task{task_id}/variation{vari}.json', 'w') as json_file:
        json.dump(raw_data, json_file, indent=4)