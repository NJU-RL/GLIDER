import deepspeed
import pandas as pd
import random
import torch
from util.model import Policy
from alg.bc import Agent
from scienceworld import ScienceWorldEnv
import copy
from prompt.inst import high_prompt, low_prompt,subtask_complete_prompt
from util.extract import extract_action_done

class EvalAgent:
    def __init__(self, args):
        self.args = args
        hierarcy_policy = Policy(args)
        self.engine, _ , _, _ = deepspeed.initialize(model=hierarcy_policy,
                                                     model_parameters=hierarcy_policy.parameters(),
                                                     config=args["ds_config"])
        self.checkpoint_dir = f"{args['check_path']}/{args['benchmark']}/{args['alg_name']}/{args['model_name']}"
        self.eval_env = ScienceWorldEnv("", envStepLimit=args['env_step_limit'])
        self.task_names = self.eval_env.getTaskNames()

    def load_policy(self, path):
        Agent.load_policy(self, path)

    def eval(self, dev_or_test):
        self.load_policy(self.checkpoint_dir)
        vari_nums = pd.read_csv(f"env/{self.args['benchmark']}/task_nums.csv",encoding='utf-8')[f'{dev_or_test}'].tolist()
        task_score = {}
        for task_id, vari_nums in enumerate(vari_nums):
            task_name = self.task_names[task_id]
            self.eval_env.load(task_name)
            task_score[task_name] = []
            if dev_or_test == "test":
                vari_ids = self.eval_env.getVariationsTest()
            elif dev_or_test == "dev":
                vari_ids = self.eval_env.getVariationsDev()
            else:
                vari_ids = list(range(vari_nums))
            
            for vari_id in random.sample(vari_ids, vari_nums):
                score = self.eval_policy(task_id, vari_id)
                # task_reward[task_name].append(reward)
                task_score[task_name].append(score)
            print("task_score: ", task_score)


        average_score= []
        for _, value in task_score.items():
            if len(value):
                average_score.append(sum(value)/len(value))
        print("result: ", sum(average_score)/len(average_score))

    def eval_policy(self, task_id, vari_id):
        episode_steps = 0
        task_name = self.task_names[task_id]
        self.eval_env.load(task_name, vari_id)
        obs, _= self.eval_env.reset()
        task_description = self.eval_env.taskdescription()
        print(f"task:{task_name}, vari:{vari_id}, {task_description}")
        # high: (high_prompt, task_description, cache_obs0, subtask_0...cache_obst)->subtask_t
        high_traj_token = self.engine.tokenizer(high_prompt + " " + task_description, return_tensors='pt')
        traj_subtask, traj_group_action = [], []
        group_action = []
        done = False
        while not done:
            state = f"Group action: {group_action}. Current observation: {obs}"
            state_token = self.engine.tokenizer(state, return_tensors='pt')
            high_traj_token["input_ids"] = torch.cat([high_traj_token["input_ids"], state_token["input_ids"]], dim = 1)
            high_traj_token["attention_mask"] = torch.cat([high_traj_token["attention_mask"], state_token["attention_mask"]], dim = 1)
            subtask = self.engine.generate_action(copy.deepcopy(high_traj_token))[0]
            print("subtask:", subtask)
            subtask_token = self.engine.tokenizer(subtask + self.engine.tokenizer.eos_token, return_tensors='pt')
            traj_subtask.append(subtask)
            high_traj_token["input_ids"] = torch.cat([high_traj_token["input_ids"], subtask_token["input_ids"]], dim = 1)
            high_traj_token["attention_mask"] = torch.cat([high_traj_token["attention_mask"], subtask_token["attention_mask"]], dim = 1)

            low_group_token = self.engine.tokenizer(low_prompt + " Subtask: " + subtask, return_tensors='pt')
            subtask_done = False
            group_action = []
            raw_action_list = []
            group_reward, group_score = 0.0, 0.0
            
            while not subtask_done:
                episode_steps += 1
                # (low_prompt, s0) -> a0, (prompt, s0, a0,..., st) -> at
                obs_token = self.engine.tokenizer("Obs: "+obs, return_tensors='pt')
                low_group_token["input_ids"] = torch.cat([low_group_token["input_ids"], obs_token["input_ids"]], dim = 1)
                low_group_token["attention_mask"] = torch.cat([low_group_token["attention_mask"], obs_token["attention_mask"]], dim = 1)
                raw_action = self.engine.generate_action(copy.deepcopy(low_group_token))[0]
                raw_action_list.append(raw_action)
                action, subtask_done = extract_action_done(raw_action)
                group_action.append(action)
                action_token = self.engine.tokenizer(raw_action+self.engine.tokenizer.eos_token, return_tensors='pt')
                low_group_token["input_ids"] = torch.cat([low_group_token["input_ids"], action_token["input_ids"]], dim = 1)
                low_group_token["attention_mask"] = torch.cat([low_group_token["attention_mask"], action_token["attention_mask"]], dim = 1)
                obs_, reward, done, info = self.eval_env.step(action)
                group_reward += reward
                group_score += info['score']
                obs = obs_
                if episode_steps == self.args['env_step_limit']:
                    done = True
                    break
            traj_group_action.append(group_action)
            print("group action: ", raw_action_list)

        # print("subtask: ", traj_subtask)
        # print("group action:", traj_group_action)
        score = max(0, info['score'])
        print(f"score: {score}")
        return score
    
    def data_collect(self, task_id, vari_id, high_data_container, low_data_container):
        """
        Args:
            high_data_container:{
                'task_description': [task_num,],
                'obs': [task_num, groups+1],
                'subtask': [task_num, groups],
                'reward': [task_num, groups],
                'score': [task_num, groups],
                'done': [task_num, groups]
            }
            low_data_container:{
                'subtask':[subtask_nums, steps],
                'obs':[subtask_nums, steps+1],
                'action':[subtask_nums, steps],
                'reward':[subtask_nums, steps],
                'score':[subtask_nums, steps],
                'done':[subtask_nums, steps]
            }
            score_threshold:[min, max]
        """
        high_obs_traj, high_subtask_traj, high_reward_traj, high_score_traj, high_done_traj = [], [], [], [], []
        episode_steps = 0
        task_name = self.task_names[task_id]
        self.eval_env.load(task_name, vari_id)
        task_description = self.eval_env.taskdescription()
        print(task_id, vari_id, task_description)
        obs, _= self.eval_env.reset()
        high_traj_token = self.engine.tokenizer(high_prompt + " " + task_description, return_tensors='pt')

        done = False
        group_action = []
        while not done:
            state = f"Group action: {group_action}. Current observation: {obs}"
            state_token = self.engine.tokenizer(state, return_tensors='pt')
            high_obs_traj.append(state)
            high_traj_token["input_ids"] = torch.cat([high_traj_token["input_ids"], state_token["input_ids"]], dim = 1)
            high_traj_token["attention_mask"] = torch.cat([high_traj_token["attention_mask"], state_token["attention_mask"]], dim = 1)
            subtask = self.engine.generate_action(copy.deepcopy(high_traj_token))[0]
            subtask_token = self.engine.tokenizer(subtask + self.engine.tokenizer.eos_token, return_tensors='pt')
            print("subtask:", subtask)
            high_subtask_traj.append(subtask)
            high_traj_token["input_ids"] = torch.cat([high_traj_token["input_ids"], subtask_token["input_ids"]], dim = 1)
            high_traj_token["attention_mask"] = torch.cat([high_traj_token["attention_mask"], subtask_token["attention_mask"]], dim = 1)

            low_group_token = self.engine.tokenizer(low_prompt + " Subtask: " + subtask, return_tensors='pt')
            subtask_done = False
            group_action = []
            group_reward, group_score = 0.0, 0.0
            raw_action_list = []
            low_obs_traj, low_reward_traj, low_score_traj, low_done_traj = [], [], [], []
            low_init_obs = obs
            while not subtask_done:
                low_obs_traj.append("Obs: "+obs)
                episode_steps += 1
                # (low_prompt, s0) -> a0, (prompt, s0, a0,..., st) -> at
                obs_token = self.engine.tokenizer("Obs: "+obs, return_tensors='pt')
                low_group_token["input_ids"] = torch.cat([low_group_token["input_ids"], obs_token["input_ids"]], dim = 1)
                low_group_token["attention_mask"] = torch.cat([low_group_token["attention_mask"], obs_token["attention_mask"]], dim = 1)
                raw_action = self.engine.generate_action(copy.deepcopy(low_group_token))[0]
                raw_action_list.append(raw_action)
                action, subtask_done = extract_action_done(raw_action)
                low_done_traj.append(subtask_done)
                group_action.append(action)
                action_token = self.engine.tokenizer(raw_action+self.engine.tokenizer.eos_token, return_tensors='pt')
                low_group_token["input_ids"] = torch.cat([low_group_token["input_ids"], action_token["input_ids"]], dim = 1)
                low_group_token["attention_mask"] = torch.cat([low_group_token["attention_mask"], action_token["attention_mask"]], dim = 1)
                obs_, reward, done, info = self.eval_env.step(action)
                group_reward += reward/100
                group_score += info['score']/100
                obs = obs_
                if episode_steps == self.args['env_step_limit']:
                    done = True
                    break
            print("group action: ", raw_action_list, info['score'])
            # is_subtask_complete_prompt = subtask_complete_prompt.replace("[subtask]",subtask)\
            #                                              .replace("[initial_obs]", low_init_obs)\
            #                                              .replace("[final_obs]", obs)\
            #                                              .replace("[action_sequence]", str(group_action))
            # subtask_complete_token = self.engine.tokenizer(is_subtask_complete_prompt, return_tensors='pt')
            # subtask_complete = self.engine.generate_action(subtask_complete_token)
            # print("subtask_complete:",subtask_complete)
            
            low_obs_traj.append("Obs: "+obs)
            low_data_container['subtask'].append(low_prompt + " Subtask: " + subtask)
            low_data_container['obs'].append(low_obs_traj)
            low_data_container['action'].append(raw_action_list)
            low_data_container['done'].append(low_done_traj)
            
            high_reward_traj.append(group_reward)
            high_score_traj.append(group_score)
            high_done_traj.append(False if episode_steps==self.args['env_step_limit'] else done)
        state = f"Group action: {group_action}. Current observation: {obs}"
        high_obs_traj.append(state)
        high_data_container['task_description'].append(high_prompt + " " + task_description)
        high_data_container['obs'].append(high_obs_traj)
        high_data_container['subtask'].append(high_subtask_traj)
        high_data_container['done'].append(high_done_traj)
        high_data_container['score'].append(high_score_traj)
        high_data_container['reward'].append(high_reward_traj)


    