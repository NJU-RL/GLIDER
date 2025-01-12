from alg.glider_awac import ActorCritic
import deepspeed
from alg.bc import Agent as BC_AGENT
from alg.glider_awac import GLIDER as GLIDER_AWAC
from scienceworld import ScienceWorldEnv
from util.replay_buffer import OnlineDataset, batch_traj_process
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torch.distributed as dist
from prompt.inst import high_prompt, low_prompt
from util.extract import extract_action_done
import copy
import pandas as pd
import random
import os


class GLIDER_ONLINE:
    def __init__(self, args):
        self.args = args
        actor_critic = ActorCritic(args)
        self.engine, _ , _, _ = deepspeed.initialize(
                                           model=actor_critic,
                                           model_parameters=[{"params": [p for p in actor_critic.base.parameters() if p.requires_grad], 
                                                              "lr": args["actor_lr"]},
                                                              {"params": [p for p in actor_critic.critic.parameters() if p.requires_grad], 
                                                               "lr": args["critic_lr"]}],
                                           config=args["ds_config"])
        path = f"{args['check_path']}/{args['benchmark']}/glider_awac/{args['model_name']}"
        BC_AGENT.load_policy(self, path)
        self.load_critic(path)
        

        self.env = ScienceWorldEnv("", envStepLimit=args['env_step_limit'])
        self.task_names = self.env.getTaskNames()

        self.loss_fct = torch.nn.MSELoss()
        self.buffer = OnlineDataset(args)

        if self.engine.global_rank == 0:
            log_dir = f"{args['log_path']}/{args['benchmark']}/{args['alg_name']}/{args['model_name']}"
            self.writer = SummaryWriter(log_dir=log_dir)
        
        self.global_step = torch.tensor(0, dtype=torch.int64).to(self.engine.device)
        self.checkpoint_dir = f"{args['check_path']}/{args['benchmark']}/{args['alg_name']}/{args['model_name']}"
        self.train_vari_list = pd.read_csv(f"env/{args['benchmark']}/task_nums.csv",encoding='utf-8')['train'].tolist()

    def train_in_domain(self,):
        batch_size_per_gpu = self.engine.train_micro_batch_size_per_gpu()
        torch.manual_seed(self.args['seed'] + self.engine.global_rank)
        
        for _ in range(self.args['episodes']):
            for task_id, vari_num in enumerate(self.train_vari_list):
                if vari_num == 0:
                    continue
                vari_id = random.choice(range(0, vari_num))                
                self.run_episode(task_id, vari_id)
                if self.buffer.ready():
                    expert_q_loss, expert_actor_loss = self.update_ac(self.buffer.sample(batch_size_per_gpu, 'high'))
                    online_q_loss, online_actor_loss = self.update_ac(self.buffer.sample(batch_size_per_gpu, 'online'))
                    low_loss = self.update_low_policy(self.buffer.sample(batch_size_per_gpu, 'low'))
                    self.engine.soft_update_target_critic(tau=self.args['tau'])
                    local_step = torch.tensor(1, dtype=torch.int64).to(self.engine.device)
                    dist.all_reduce(local_step, op=dist.ReduceOp.SUM) 
                    if self.engine.local_rank == 0:  # Only log on the main process
                        print(f"step:{self.global_step.item()}; low_loss:{low_loss.item()};")
                        print(f"expert_actor_loss,expert_q_loss:{expert_actor_loss.item(),expert_q_loss.item()};")
                        print(f"online_actor_loss, online_q_loss:{online_actor_loss.item(), online_q_loss.item()}")
                        self.writer.add_scalar('Loss/expert/q_loss', expert_q_loss.item(), self.global_step.item())
                        self.writer.add_scalar('Loss/expert/actor_loss', expert_actor_loss.item(), self.global_step.item())
                        self.writer.add_scalar('Loss/online/q_loss', online_q_loss.item(), self.global_step.item())
                        self.writer.add_scalar('Loss/online/actor_loss', online_actor_loss.item(), self.global_step.item())
                    if self.global_step.item() % self.args['eval_freq'] == 0:
                        BC_AGENT.save_policy(self)  
                    self.global_step += local_step
        BC_AGENT.save_policy(self)

    def update_low_policy(self,batch_data):
        batch_low_tokens = batch_traj_process(batch_data['subtask'],
                                             batch_data['obs'],
                                             batch_data['action'],
                                             self.engine.tokenizer).to(self.engine.device)
        low_log_probs, low_masks = self.engine.get_log_prob(batch_low_tokens)
        low_valid_log_prob = BC_AGENT.extract_valid_action_probs(self, low_log_probs, low_masks,
                                                                 max(batch_low_tokens['action_end_mask'].sum(dim=1)))
        low_loss = -low_valid_log_prob.mean()
        self.engine.backward(low_loss)
        self.engine.step()
        return low_loss

    def update_ac(self, batch_data):
        batch_tokens = batch_traj_process(batch_data['task_description'],
                                          batch_data['obs'],
                                          batch_data['subtask'],
                                          self.engine.tokenizer).to(self.engine.device)
        rewards, dones = self.prepare_tensor(batch_data['reward'], batch_data['done'])

        # Critic
        with torch.no_grad():
            hidden_states, _, action_end_mask = self.engine.get_hidden_states(batch_tokens)
            target_qsa = self.engine.target_critic_forward(hidden_states) # (batch, seq_len)
            target_qsa, _ = GLIDER_AWAC.extract_valid(self, target_qsa, action_end_mask) # (batch, num_action)

        q_sa = self.engine.critic_forward(hidden_states)
        q_sa, _ = GLIDER_AWAC.extract_valid(self, q_sa, action_end_mask) # (batch, num_action)

        # L_Q
        target = rewards + (1-dones) * F.pad(target_qsa[:, 1:], (0, 1), value=0) * self.args['gama']    # (batch, num_action)
        q_loss = self.loss_fct(q_sa, target)
        self.engine.backward(q_loss)
        self.engine.step()
        
        # Actor
        action_log_probs, action_masks = self.engine.get_log_prob(batch_tokens)
        # valid_action_log_probs,_ = self.extract_valid(action_log_probs, action_end_mask[:, 1:])
        valid_action_log_probs = BC_AGENT.extract_valid_action_probs(self, action_log_probs, action_masks, q_sa.size(1))
       
        actor_loss = -torch.mean(target_qsa.detach()*valid_action_log_probs)
        self.engine.backward(actor_loss)
        self.engine.step()
        return q_loss, actor_loss
    
    def prepare_tensor(self, rewards, dones):
        """
        Args:
            rewards: List[(batch, steps)]
            dones: List[(batch, steps)]
        Returns:
            reward: tensor -> (batch, padding_steps)
            dones: tensor -> (batch, padding_steps)
        """
        # process reward
        reward_list = [torch.tensor(seq, dtype=torch.float, device=self.engine.device) for seq in rewards]
        done_list = [torch.tensor(seq, dtype=torch.float, device=self.engine.device) for seq in dones]
        
        # padding
        reward_tensor = pad_sequence(reward_list, batch_first=True, padding_value=0.0)
        done_tensor = pad_sequence(done_list, batch_first=True, padding_value=0)
        
        return reward_tensor, done_tensor
    
                    

    def run_episode(self, task_id, vari_id):
        episode_steps = 0
        traj_dict = {key:[] for key in ['obs', 'subtask', 'reward', 'score', 'done']}
        task_name = self.task_names[task_id]
        self.env.load(task_name, vari_id)
        task_description = self.env.taskdescription()
        traj_dict['task_description'] = high_prompt + " " + task_description
        high_traj_token = self.engine.tokenizer(traj_dict['task_description'], return_tensors='pt')
        obs, _= self.env.reset()
        group_action = []
        done = False
        while not done:
            state = f"Group action: {group_action}. Current observation: {obs}"
            state_token = self.engine.tokenizer(state, return_tensors='pt')
            traj_dict['obs'].append(state)
            high_traj_token["input_ids"] = torch.cat([high_traj_token["input_ids"], state_token["input_ids"]], dim = 1)
            high_traj_token["attention_mask"] = torch.cat([high_traj_token["attention_mask"], state_token["attention_mask"]], dim = 1)
            subtask = self.engine.generate_action(copy.deepcopy(high_traj_token))[0]
            traj_dict['subtask'].append(subtask)
            subtask_token = self.engine.tokenizer(subtask + self.engine.tokenizer.eos_token, return_tensors='pt')
            high_traj_token["input_ids"] = torch.cat([high_traj_token["input_ids"], subtask_token["input_ids"]], dim = 1)
            high_traj_token["attention_mask"] = torch.cat([high_traj_token["attention_mask"], subtask_token["attention_mask"]], dim = 1)

            low_group_token = self.engine.tokenizer(low_prompt + " Subtask: " + subtask, return_tensors='pt')
            subtask_done = False
            group_action = []
            raw_action_list = []
            group_reward, group_score = 0.0, 0.0

            while not subtask_done:
                episode_steps += 1
                obs_token = self.engine.tokenizer(" Obs: "+obs, return_tensors='pt')
                low_group_token["input_ids"] = torch.cat([low_group_token["input_ids"], obs_token["input_ids"]], dim = 1)
                low_group_token["attention_mask"] = torch.cat([low_group_token["attention_mask"], obs_token["attention_mask"]], dim = 1)
                raw_action = self.engine.generate_action(copy.deepcopy(low_group_token))[0]
                raw_action_list.append(raw_action)
                action, subtask_done = extract_action_done(raw_action)
                group_action.append(action)
                action_token = self.engine.tokenizer(raw_action+self.engine.tokenizer.eos_token, return_tensors='pt')
                low_group_token["input_ids"] = torch.cat([low_group_token["input_ids"], action_token["input_ids"]], dim = 1)
                low_group_token["attention_mask"] = torch.cat([low_group_token["attention_mask"], action_token["attention_mask"]], dim = 1)
                obs_, reward, done, info = self.env.step(action)
                group_reward += reward/100
                group_score += info['score']/100
                obs = obs_
                if episode_steps == self.args['env_step_limit']:
                    done = True
                    break
            traj_dict['score'].append(group_score)
            traj_dict['reward'].append(group_reward)
            traj_dict['done'].append(done)
            
        state = f"Group action: {group_action}. Current observation: {obs}"
        traj_dict['obs'].append(state)
        self.buffer.push(traj_dict)


    def load_critic(self, path):
        critic_path = os.path.join(path, "critic.pth")
        if os.path.exists(critic_path):
            self.engine.module.critic.load_state_dict(torch.load(critic_path,weights_only=True))
            self.engine.soft_update_target_critic(tau=1.0)
        else:
            print(f"No checkpoint found at {critic_path}")

        
            
