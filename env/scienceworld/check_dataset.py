from scienceworld import ScienceWorldEnv
import json
import os


TASK_NUM = 30
env = ScienceWorldEnv("", envStepLimit=1000)
task_names = env.getTaskNames()


for task_id in [10]:
    env.load(task_names[task_id], generateGoldPath=False)
    vari_ids = env.getVariationsTrain()
    for vari in vari_ids:
        obs_traj, next_obs_traj, score_traj, reward_traj, done_traj = [],[],[],[],[]
        env.load(task_names[task_id], vari)
        with open(f'dataset/scienceworld/task{task_id}/variation{vari}.json', 'r') as json_file:
            raw_data = json.load(json_file)
        if "group_action" in raw_data.keys():
            action_traj = []
            for group in raw_data['group_action']:
                for a in group:
                    action_traj.append(a)
        else:
            action_traj = raw_data['action']
        obs, _ = env.reset()
        for action in action_traj:
            obs_traj.append(obs)
            obs_, reward, isCompleted, infos = env.step(action)
            next_obs_traj.append(obs_)
            obs = obs_
            reward_traj.append(reward/100)
            score_traj.append(infos['score']/100)
            done_traj.append(isCompleted)
        if not isCompleted:
            print(task_id, vari, "error")
            print(score_traj)
        raw_data['obs'] = obs_traj
        raw_data['next_obs'] = next_obs_traj
        raw_data['score'] = score_traj
        raw_data['reward'] = reward_traj
        raw_data['done'] = done_traj
        # print(raw_data)
        # with open(f'dataset/scienceworld/task{task_id}/variation{vari}.json', 'w') as json_file:
        #     json.dump(raw_data, json_file, indent=4)
        # exit()

