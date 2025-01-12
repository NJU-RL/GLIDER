import json

task_id = 11
for vari in range(62):
    with open(f'dataset/scienceworld/task{task_id}/variation{vari}.json', 'r') as json_file:
        raw_data = json.load(json_file)
    subtask = []
    action_group = []
    for i in range(len(raw_data['subtask'])):
        if len(raw_data['group_action'][i]) != 0:
            subtask.append(raw_data['subtask'][i])
            action_group.append(raw_data['group_action'][i])

    raw_data["group_action"] = action_group
    raw_data['subtask'] = subtask

    print(raw_data)
    with open(f'dataset/scienceworld/task{task_id}/variation{vari}.json', 'w') as json_file:
        json.dump(raw_data, json_file, indent=4)