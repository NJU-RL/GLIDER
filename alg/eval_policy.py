
import deepspeed
import pandas as pd
import random
import torch
from util.model import Policy
from alg.bc import Agent
from scienceworld import ScienceWorldEnv
import copy
import json

class EvalAgent:
    def __init__(self, args):
        self.args = args
        policy = Policy(args)
        self.engine, _ , _, _ = deepspeed.initialize(model=policy,
                                                    model_parameters=[p for p in policy.parameters() if p.requires_grad],
                                                    config=args["ds_config"])
        self.checkpoint_dir = f"{args['check_path']}/{args['benchmark']}/{args['alg_name']}/{args['model_name']}"
        Agent.load_policy(self, self.checkpoint_dir)  # load model

        self.eval_env = ScienceWorldEnv("", envStepLimit=args['env_step_limit'])
        self.task_names = self.eval_env.getTaskNames()

    def data_collect(self, task_id, vari_id, data_container, score_threshold):
        """
        Args:
            data_container:{
                'task_description': [],
                'obs': [],
                'action': [],
                'next_obs': [],
                'reward': [],
                'score': [],
                'done': []
            }
            score_threshold:[min, max]
        """
        obs_traj, action_traj, next_obs_traj, reward_traj, score_traj, done_traj = [],[],[],[],[],[]
        task_name = self.task_names[task_id]
        self.eval_env.load(task_name, vari_id)
        task_description = self.eval_env.taskdescription()
        
        traj_token = self.engine.tokenizer(task_description, return_tensors='pt')
        
        obs, _ = self.eval_env.reset()
        obs_token = self.engine.tokenizer(obs, return_tensors='pt')

        done, episode_steps =False, 0
        while not done:
            episode_steps += 1
            # (prompt, s0) -> a0, (prompt, s0, a0,..., st) -> at
            obs_traj.append(obs)
            obs_token = self.engine.tokenizer(obs, return_tensors='pt')
            traj_token["input_ids"] = torch.cat([traj_token["input_ids"], obs_token["input_ids"]], dim = 1)
            traj_token["attention_mask"] = torch.cat([traj_token["attention_mask"], obs_token["attention_mask"]], dim = 1)
            
            action = self.engine.generate_action(copy.deepcopy(traj_token))[0]
            action_token = self.engine.tokenizer(action+self.engine.tokenizer.eos_token, return_tensors='pt')
            action_traj.append(action)
            traj_token["input_ids"] = torch.cat([traj_token["input_ids"], action_token["input_ids"]], dim = 1)
            traj_token["attention_mask"] = torch.cat([traj_token["attention_mask"], action_token["attention_mask"]], dim = 1)
            
            obs_, reward, done, info = self.eval_env.step(action)
            reward = reward/100
            score = info['score']/100
            next_obs_traj.append(obs_)
            reward_traj.append(reward)
            score_traj.append(score)
            done_traj.append(done)            
            obs = obs_
            if episode_steps == self.args['env_step_limit']:
                done = True

        if score_threshold[0] <= score <= score_threshold[1]:
            data_container['task_description'].append(task_description)
            data_container['obs'].append(obs_traj)
            data_container['action'].append(action_traj)
            data_container['next_obs'].append(next_obs_traj)
            data_container['reward'].append(reward_traj)
            data_container['score'].append(score_traj)
            data_container['done'].append(done_traj)
            


    def eval_policy(self, dev_or_test):
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
                self.eval_env.load(task_name, vari_id)
                obs, _ = self.eval_env.reset()
                task_description = self.eval_env.taskdescription()
                traj_token = self.engine.tokenizer(task_description, return_tensors='pt')
                
                obs_token = self.engine.tokenizer(obs, return_tensors='pt')
                action_traj, done, episode_steps = [], False, 0
                while not done:
                    episode_steps += 1
                    # (prompt, s0) -> a0, (prompt, s0, a0,..., st) -> at
                    obs_token = self.engine.tokenizer(obs, return_tensors='pt')
                    traj_token["input_ids"] = torch.cat([traj_token["input_ids"], obs_token["input_ids"]], dim = 1)
                    traj_token["attention_mask"] = torch.cat([traj_token["attention_mask"], obs_token["attention_mask"]], dim = 1)
                    # print(f"**{episode_steps}**")
                    # print(self.engine.tokenizer.batch_decode(traj_token["input_ids"]))
                    action = self.engine.generate_action(copy.deepcopy(traj_token))[0]
                    action_token = self.engine.tokenizer(action+self.engine.tokenizer.eos_token, return_tensors='pt')
                    traj_token["input_ids"] = torch.cat([traj_token["input_ids"], action_token["input_ids"]], dim = 1)
                    traj_token["attention_mask"] = torch.cat([traj_token["attention_mask"], action_token["attention_mask"]], dim = 1)
                    
                    # if episode_steps ==4:
                    #     exit()
                    obs_, reward, done, info = self.eval_env.step(action)
                    action_traj.append(action)
                    # print(f"{task_name}; step:{episode_steps}; obs:{obs}; action:{action}, reward:{reward}, score:{info['score']} ")
                    
                    obs = obs_
                    if episode_steps == self.args['env_step_limit']:
                        done = True
                     
                print(f"{task_name}; {action_traj}; {max(0, info['score'])}")
                # task_reward[task_name].append(reward)
                task_score[task_name].append(max(0, info['score']))
            print("task_score: ", task_score)

        average_score= []
        for _, value in task_score.items():
            if len(value):
                average_score.append(sum(value)/len(value))
        print("result: ", sum(average_score)/len(average_score))


