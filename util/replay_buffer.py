from torch.utils.data import Dataset
# import os
import json
import pandas as pd
import torch
from transformers.tokenization_utils_base import BatchEncoding
from prompt.inst import high_prompt, low_prompt
from collections import deque
import random

def batch_traj_process(batch_prompt, batch_states, batch_actions, tokenizer):   # test done
        batch_input, batch_mask, batch_labels = [], [], []
        batch_state_end_mask, batch_action_end_mask = [], []
        for prompt, states, actions in zip(batch_prompt, batch_states, batch_actions):
            prompt_token = tokenizer(prompt, return_tensors='pt')
            input_tensors = [prompt_token['input_ids']]
            mask_tensors = [prompt_token['attention_mask']]
            labels = [torch.full_like(input_tensors[0], -100)]

            state_end_masks = [torch.zeros_like(input_tensors[0])]
            action_end_masks = [torch.zeros_like(input_tensors[0])]

            for state, action in zip(states[:-1], actions):
                action = action + tokenizer.eos_token
                state_token = tokenizer(state, return_tensors='pt')
                action_token = tokenizer(action, return_tensors='pt')

                state_label = torch.full_like(state_token['input_ids'], -100)
                action_label = action_token['input_ids']

                # record state and action end token position
                state_end_mask = torch.zeros_like(state_token['input_ids'])
                action_end_mask = torch.zeros_like(action_token['input_ids'])
                state_end_mask[0, -1], action_end_mask[0, -1] = 1, 1 # state and action end token mark as 1

                input_tensors.extend([state_token['input_ids'], action_token['input_ids']])
                mask_tensors.extend([state_token['attention_mask'], action_token['attention_mask']])
                labels.extend([state_label, action_label])
                state_end_masks.extend([state_end_mask, torch.zeros_like(action_token['input_ids'])])
                action_end_masks.extend([torch.zeros_like(state_token['input_ids']), action_end_mask])
            
            final_state_token = tokenizer(states[-1], return_tensors='pt')
            final_state_label = torch.full_like(final_state_token['input_ids'], -100)
            final_state_end_mask = torch.zeros_like(final_state_token['input_ids'])
            final_state_end_mask[0, -1] = 1  # Marks the end position of the last state
            
            input_tensors.append(final_state_token['input_ids'])
            mask_tensors.append(final_state_token['attention_mask'])
            labels.append(final_state_label)
            state_end_masks.append(final_state_end_mask)
            action_end_masks.append(torch.zeros_like(final_state_token['input_ids']))

            batch_input.append(torch.cat(input_tensors, dim=1))
            batch_mask.append(torch.cat(mask_tensors, dim=1))
            batch_labels.append(torch.cat(labels, dim=1))

            batch_state_end_mask.append(torch.cat(state_end_masks, dim=1))
            batch_action_end_mask.append(torch.cat(action_end_masks, dim=1))

        # Padding
        padded_input = torch.nn.utils.rnn.pad_sequence([x.squeeze(0) for x in batch_input], 
                                                    batch_first=True, 
                                                    padding_value=tokenizer.pad_token_id)
        padded_mask = torch.nn.utils.rnn.pad_sequence([x.squeeze(0) for x in batch_mask], 
                                                    batch_first=True, 
                                                    padding_value=0)
        padded_labels = torch.nn.utils.rnn.pad_sequence([x.squeeze(0) for x in batch_labels], 
                                                    batch_first=True, 
                                                    padding_value=-100)
        
        padded_state_end_mask = torch.nn.utils.rnn.pad_sequence([x.squeeze(0) for x in batch_state_end_mask],
                                                                 batch_first=True,
                                                                   padding_value=0)
        padded_action_end_mask = torch.nn.utils.rnn.pad_sequence([x.squeeze(0) for x in batch_action_end_mask], 
                                                                 batch_first=True, 
                                                                 padding_value=0)
    
        
        return BatchEncoding({
            "input_ids": padded_input,
            "attention_mask": padded_mask,
            "labels": padded_labels,
            "state_end_mask": padded_state_end_mask,
            "action_end_mask": padded_action_end_mask
        })


class SequenceDataset(Dataset):
    def __init__(self, args):
        super(SequenceDataset, self).__init__()
        self.args = args
        self.data = {
            "task_description":[],
            "obs": [],
            "action": [],
            "next_obs": [],
            "reward": [],
            "score": [],
            "done": []
        }
        self.load_data()
    
    def load_data(self):
        vari_nums = pd.read_csv(f"env/{self.args['benchmark']}/task_nums.csv",encoding='utf-8')['train'].tolist()
        for task_id, vari_num in enumerate(vari_nums):
            for vari_id in range(vari_num):
                path = f"dataset/{self.args['benchmark']}/task{task_id}/variation{vari_id}.json"
                with open(path, 'r') as f:
                    raw_traj = json.load(f)
                for key in self.data.keys():
                    self.data[key].append(raw_traj[key])
        

    def __len__(self):
        return len(self.data["obs"])
    
    def __getitem__(self, idx):
        return {key: value[idx] for key, value in self.data.items()}
    
    @staticmethod
    def collate_fn(batch):
        # Custom collate_fn to handle sequences of different lengths
        batch_data = {
            'task_description': [],
            'obs': [],
            'action': [],
            'next_obs': [],
            'reward': [],
            'score': [],
            'done': []
        }
        
        for sample in batch:
            for key in batch_data:
                batch_data[key].append(sample[key])
                
        return batch_data

class OnlineDataset:
    def __init__(self, args):
        self.args = args
        with open(f"dataset/{self.args['benchmark']}/high_data/expert.json", 'r') as f:
            self.high_data = json.load(f)
        with open(f"dataset/{self.args['benchmark']}/low_data/expert.json", 'r') as f:
            self.low_data = json.load(f)
        self.low_len = len(self.low_data['subtask'])
        self.high_len = len(self.high_data['task_description'])
        self.data_size = max(self.low_len,self.high_len)
        self.online_data = {key:deque(maxlen=self.args['online_data_size']) for key in self.high_data.keys()}
        
    def push(self, traj_dict):
        for key in self.online_data.keys():
            self.online_data[key].append(traj_dict[key])

    def ready(self):
        first_key = list(self.online_data.keys())[0]
        return len(self.online_data[first_key]) == self.args['online_data_size']
   
    def sample(self, batch_size):
        data = {k: list(v) for k, v in self.online_data.items()}
        data_length = len(data[list(data.keys())[0]])
        indices = random.sample(range(data_length), min(batch_size, data_length))
        batch = {}
        for key in data.keys():
            batch[key] = [data[key][i] for i in indices]
        return batch
    
    def __len__(self):
        return self.data_size
    
    def __getitem__(self, idx):
        high_idx = idx % self.high_len
        low_idx = idx % self.low_len
        return {
                'high':{key: value[high_idx] for key, value in self.high_data.items()},
                'low':{key: value[low_idx] for key, value in self.low_data.items()}
               }
    
    @staticmethod
    def collate_fn(batch):
        batch_data = {
            'high': {key: [] for key in batch[0]['high'].keys()},
            'low': {key: [] for key in batch[0]['low'].keys()}
        }

        for sample in batch:
            for key in batch_data['high']:
                batch_data['high'][key].append(sample['high'][key])
            for key in batch_data['low']:
                batch_data['low'][key].append(sample['low'][key])
        return batch_data


class HierarchyDataset(Dataset):
    def __init__(self, args):
        super(HierarchyDataset, self).__init__()
        self.args = args
        self.load_data()

    def load_data(self):
        if self.args['mode']=='collect':
            with open(f"dataset/high_data_half{self.args['half']}.json", 'r') as f:
                self.high_data = json.load(f)
        else:
            with open(f"dataset/{self.args['benchmark']}/high_data/expert.json", 'r') as f:
                self.high_data = json.load(f)
        if self.args['mode'] == 'rl':
            self.medium_data = {key:[] for key in self.high_data.keys()}
            for path in self.args['medium_dataset']:
                with open(f"dataset/{self.args['benchmark']}/high_data/{path}", 'r') as f:
                    medium = json.load(f)
                for key in self.high_data:
                    self.medium_data[key] += medium[key]

        with open(f"dataset/{self.args['benchmark']}/low_data/expert.json", 'r') as f:
            self.low_data = json.load(f)
        
        self.low_len = len(self.low_data['subtask'])
        self.high_len = len(self.high_data['task_description'])
        self.data_size = max(self.low_len,self.high_len)
        if self.args['mode'] == 'rl':
            self.medium_len = len(self.medium_data['task_description'])
            self.data_size = max(self.data_size, self.medium_len)
            print(self.high_len, self.low_len, self.medium_len)
        print(self.high_len, self.low_len)
        # repeat_times = low_length // high_length
        # remainder = low_length % high_length

        # for key in self.high_data:
        #     self.high_data[key] = self.high_data[key] * repeat_times + self.high_data[key][:remainder]
        # print(len(self.high_data['task_description']), len(self.low_data['subtask']))

    def __len__(self):
        return self.data_size
    
    def __getitem__(self, idx):
        high_idx = idx % self.high_len
        low_idx = idx % self.low_len
        if self.args['mode'] == 'rl':
            medium_idx = idx % self.medium_len
            return {
                'high':{key: value[high_idx] for key, value in self.high_data.items()},
                'low':{key: value[low_idx] for key, value in self.low_data.items()},
                'medium':{key: value[medium_idx] for key, value in self.medium_data.items()}
            }
        else:  # "bc" or "collect"
            return {
                'high':{key: value[high_idx] for key, value in self.high_data.items()},
                'low':{key: value[low_idx] for key, value in self.low_data.items()}
            }
    
    @staticmethod
    def collate_fn(batch):
        """
        Returns:
            {
                'high': {
                    'task_description': [sample1, sample2, ...],
                    'other_key': [sample1, sample2, ...],
                    ...
                },
                'low': {
                    'subtask': [sample1, sample2, ...],
                    'other_key': [sample1, sample2, ...],
                    ...
                }
            }
        """
        batch_data = {
            'high': {key: [] for key in batch[0]['high'].keys()},
            'low': {key: [] for key in batch[0]['low'].keys()}
        }
        if 'medium' in batch[0]:
            batch_data['medium'] = {key: [] for key in batch[0]['medium'].keys()}

        for sample in batch:
            for key in batch_data['high']:
                batch_data['high'][key].append(sample['high'][key])
            for key in batch_data['low']:
                batch_data['low'][key].append(sample['low'][key])
            if 'medium' in sample:
                for key in batch_data['medium']:
                    batch_data['medium'][key].append(sample['medium'][key])

        return batch_data

    def convert_data(self):
        data = {
            "task_description":[],
            "subtask":[],
            "obs": [],
            "action": [],
            "group_action":[],
            "next_obs": [],
            "reward": [],
            "score": [],
            "done": []
        }
        low_dataset = {
            "subtask":[],
            "obs":[],
            "action":[],
            "reward":[],
            "score":[],
            "done":[]
        }
        high_dataset = {
            "task_description":[],
            "obs":[],
            "subtask":[],
            "reward":[],
            "score":[],
            "done":[]
        }
        vari_nums = pd.read_csv(f"env/{self.args['benchmark']}/task_nums.csv",encoding='utf-8')['train'].tolist()
        max_vari_nums = max(vari_nums)
        for task_id, vari_num in enumerate(vari_nums):
            if vari_num == 0:
                continue
            for vari_id in range(max_vari_nums):
                if self.args['half']==1:
                    path = f"dataset/{self.args['benchmark']}/task{task_id}/variation{vari_id%(vari_num//2)+(vari_num//2)}.json"
                elif self.args['half']==0:
                    path = f"dataset/{self.args['benchmark']}/task{task_id}/variation{vari_id%(vari_num//2)}.json"
                with open(path, 'r') as f:
                    raw_traj = json.load(f)
                    # print(raw_traj)
                for key in data.keys():
                    # print(task_id, vari_id)
                    data[key].append(raw_traj[key])

        with open('dataset/expert_traj.json', 'w') as json_file:
            json.dump(data, json_file, indent=4)
        
        high_dataset['task_description']=[high_prompt + " " + task for task in data['task_description']]
        high_dataset['subtask'] = data['subtask']
        for i in range(len(data['task_description'])):
            step = 0
            traj_high_obs, traj_high_reward, traj_high_score, traj_high_done= [],[],[],[]
            action_cache = "Group action:[]."
            print(i,data['task_description'][i])
            for group_id, group_action in enumerate(data['group_action'][i]):
                traj_high_obs.append(action_cache + " Current observation: " + data['obs'][i][step])
                group_score, group_reward = 0.0, 0.0
                
                low_dataset['subtask'].append(low_prompt + " Subtask: " + data['subtask'][i][group_id])
                low_obs_group = ["Obs: " + data['obs'][i][step]]
                for k, _ in enumerate(group_action):
                    low_obs_group.append("Obs: " + data['next_obs'][i][step])
                    # print(data['task_description'][i])
                    # print(i,len(data['next_obs'][i]), step)
                    group_score += data['score'][i][step]
                    group_reward += data['reward'][i][step]
                    step += 1
                action_cache = "Group action:" + str(group_action)
                group_done = [False]*(len(group_action)-1)+[True]
                low_dataset['done'].append(group_done)
                low_dataset['reward'].append([0.0]*(len(group_action)-1)+[1.0])
                low_dataset['score'].append([0.0]*(len(group_action)-1)+[1.0])
                low_dataset['action'].append([action+"; "+str(done) for action, done in zip(group_action, group_done)])
                low_dataset['obs'].append(low_obs_group)
                traj_high_reward.append(group_reward)
                traj_high_score.append(group_score)
                traj_high_done.append(data['done'][i][step-1])
            traj_high_obs.append(action_cache + " Current observation: " + data['next_obs'][i][step-1])   
            high_dataset['obs'].append(traj_high_obs)
            high_dataset['reward'].append(traj_high_reward)
            high_dataset['score'].append(traj_high_score)
            high_dataset['done'].append(traj_high_done)
        
        with open(f"dataset/low_data_half{self.args['half']}.json", 'w') as json_file:
            json.dump(low_dataset, json_file, indent=4)
        with open(f"dataset/high_data_half{self.args['half']}.json", 'w') as json_file:
            json.dump(high_dataset, json_file, indent=4)
        
            
        