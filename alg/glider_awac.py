from util.model import Policy, Critic
from alg.bc import Agent as BC_AGENT
from util.replay_buffer import HierarchyDataset, batch_traj_process
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torch.nn as nn
import deepspeed
import torch
import copy
import os


class ActorCritic(Policy):
    def __init__(self, args):
        super().__init__(args)
        hidden_dim = self.base.config.hidden_size
        self.critic = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, 1))
        self.target_critic = copy.deepcopy(self.critic)
        for param in self.target_critic.parameters():
            param.requires_grad = False
        self.soft_update_target_critic(tau=1.0)

    def soft_update_target_critic(self, tau):
        assert 0.0 <= tau <= 1.0
        for target_param, param in zip(self.target_critic.parameters(), 
                                       self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) 
                                    + param.data * tau)
    
    def critic_forward(self, x):
        return self.critic(x).squeeze(-1)
    
    @torch.no_grad()
    def target_critic_forward(self, x):
        return self.target_critic(x).squeeze(-1)
            
class GLIDER:
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
        
        
        path = f"{args['check_path']}/{args['benchmark']}/glider_bc/{args['model_name']}"
        BC_AGENT.load_policy(self, path)
        
        self.loss_fct = torch.nn.MSELoss()
        self.buffer = HierarchyDataset(args)

        if self.engine.global_rank == 0:
            log_dir = f"{args['log_path']}/{args['benchmark']}/{args['alg_name']}/{args['model_name']}"
            self.writer = SummaryWriter(log_dir=log_dir)

        self.global_step = torch.tensor(0, dtype=torch.int64).to(self.engine.device)
        self.checkpoint_dir = f"{args['check_path']}/{args['benchmark']}/{args['alg_name']}/{args['model_name']}"
    
    def save_critic(self):
        model_path = os.path.join(self.checkpoint_dir, "critic.pth")
        torch.save(self.engine.module.critic.state_dict(), model_path)
    
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
            target_qsa, _ = self.extract_valid(target_qsa, action_end_mask) # (batch, num_action)

        q_sa = self.engine.critic_forward(hidden_states)
        q_sa, _ = self.extract_valid(q_sa, action_end_mask) # (batch, num_action)

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
            

    def update(self,):
        batch_size_per_gpu = self.engine.train_micro_batch_size_per_gpu()
        sampler = DistributedSampler(self.buffer, 
                                     num_replicas=self.engine.world_size, 
                                     rank=self.engine.local_rank)
        
        dataloader = DataLoader(self.buffer, 
                                batch_size=batch_size_per_gpu, 
                                sampler=sampler,
                                collate_fn=HierarchyDataset.collate_fn)
        
        for epoch in range(self.args['epochs']):
            sampler.set_epoch(epoch)    # each epoch shuffle
            for batch in dataloader:
                """
                Args:
                    batch:{
                        "task_description": [traj_nums,], or "subtask":[group_nums, ]
                        "obs":[traj_nums, steps+1],       or "obs":[group_nums, steps+1]
                        "subtask":[traj_nums, steps],     or "action":[group_nums, steps]
                        "reward": [traj_nums, steps]      or "reward":[group_nums, steps]
                        "done": [traj_nums, steps]        or "done": [grop_nums, steps]
                    }
                Update Critic:
                    L_Q(\phi) = E_{s,a,r,s'~D}[Q_{\phi}(s,a) - r - \gamma * V_{\bar{\psi}(s')]
                    L_V(\psi) = E_{s~D}[E_{a~\pi_\theta(·|s)}[V_\psi(s)-Q_{\bar{\theta}}(s,a)]]
                Update Actor:
                    L_\pai(\theta) = -E_{s,a~D}[exp(1/lammda * A(s,a)) * log\pai_\theta(a|s)]
                """
                # high level
                expert_q_loss, expert_actor_loss = self.update_ac(batch['high'])
                medium_q_loss, medium_actor_loss = self.update_ac(batch['medium'])
               
                self.engine.soft_update_target_critic(tau=self.args['tau'])

                
            
                # low level
                batch_low_tokens = batch_traj_process(batch['low']['subtask'],
                                                      batch['low']['obs'],
                                                      batch['low']['action'],
                                                      self.engine.tokenizer).to(self.engine.device)
                low_log_probs, low_masks = self.engine.get_log_prob(batch_low_tokens)
                low_valid_log_prob = BC_AGENT.extract_valid_action_probs(self, low_log_probs, low_masks,
                                                                         max(batch_low_tokens['action_end_mask'].sum(dim=1)))
                low_loss = -low_valid_log_prob.mean()
                self.engine.backward(low_loss)
                self.engine.step()
                
                
                if self.engine.local_rank == 0:  # Only log on the main process
                    self.writer.add_scalar('Loss/expert/q_loss', expert_q_loss.item(), self.global_step.item())
                    self.writer.add_scalar('Loss/expert/actor_loss', expert_actor_loss.item(), self.global_step.item())
                    print(f"expert; step:{self.global_step.item()}; actor_loss:{expert_actor_loss.item()}; critic_loss:{expert_q_loss.item()}")
                    self.writer.add_scalar('Loss/medium/q_loss', medium_q_loss.item(), self.global_step.item())
                    self.writer.add_scalar('Loss/medium/actor_loss', medium_actor_loss.item(), self.global_step.item())
                    print(f"medium; step:{self.global_step.item()}; actor_loss:{medium_actor_loss.item()}; critic_loss:{medium_q_loss.item()}")
                    self.writer.add_scalar('Loss/low/actor_loss', low_loss, self.global_step.item())
                    print(f"low; step:{self.global_step.item()}; loss:{low_loss.item()}")

                if self.global_step.item() % self.args['eval_freq'] == 0:
                    BC_AGENT.save_policy(self)
                    self.save_critic()
                self.global_step += 1
        BC_AGENT.save_policy(self)
        self.save_critic()

    def get_policy_q(self, batch_prompt, batch_obs_list, batch_action_list):
        """
        Args:
            batch_obs_list: List[List[str]], shape: (batch, steps+1)
            batch_action_list: List[List[int]], shape: (batch, steps)
        Returns:
            q_values: Q(s, a~π), (batch, max_steps)
        """
        q_values = []
        for prompt, obs_list, action_list in zip(batch_prompt, batch_obs_list, batch_action_list):
            obs_list = obs_list[:-1]
            traj_len = len(obs_list)
            q_list = []
            traj_token = self.engine.tokenizer(prompt, return_tensors='pt')
            for t in range(traj_len):
                obs_token = self.engine.tokenizer(obs_list[t], return_tensors='pt')
                traj_token["input_ids"] = torch.cat([traj_token["input_ids"], obs_token["input_ids"]], dim = 1)
                traj_token["attention_mask"] = torch.cat([traj_token["attention_mask"], obs_token["attention_mask"]], dim = 1)
                
                pi_action = self.engine.generate_action(copy.deepcopy(traj_token))[0]
                pi_action_token = self.engine.tokenizer(pi_action+self.engine.tokenizer.eos_token, return_tensors='pt')
                input_token = {"input_ids": torch.cat([traj_token["input_ids"], pi_action_token["input_ids"]], dim=1).to(self.engine.device),
                               "attention_mask": torch.cat([traj_token["attention_mask"], pi_action_token["attention_mask"]], dim=1).to(self.engine.device)}
                hidden_states = self.engine.base(**input_token, output_hidden_states=True).hidden_states[-1][:,-1] # (1, hidden_dim)
                q_value, _ = self.engine.target_critic(hidden_states)
                q_list.append(q_value)

                action_token = self.engine.tokenizer(action_list[t]+self.engine.tokenizer.eos_token, return_tensors='pt')
                traj_token["input_ids"] = torch.cat([traj_token["input_ids"], action_token["input_ids"]], dim = 1)
                traj_token["attention_mask"] = torch.cat([traj_token["attention_mask"], action_token["attention_mask"]], dim = 1)

            q_values.append(torch.cat(q_list))
        q_values = pad_sequence(q_values, batch_first=True, padding_value=0.0)
        return q_values
                
                
    def extract_valid(self, value, valid_mark):
        """
        Args:
            value: (batch, seq_len)
            valid_mark: (batch, seq_len), where 1 indicates extraction position
        Returns:
            valid_value: The extracted sequence -> (batch, padding_steps)
            mask: padding position->(batch, padding_steps)(1111...00)
        """
        batch_size = value.size(0)
        max_valid_len = valid_mark.sum(dim=1).max().item()

        valid_value = torch.zeros(batch_size, max_valid_len, device=value.device)
        mask = torch.zeros(batch_size, max_valid_len, device=value.device)
        for i in range(batch_size):
            valid_idx = torch.where(valid_mark[i] == 1)[0]
            valid_len = valid_idx.size(0) # same to value and q_value
            
            valid_value[i, :valid_len] = value[i][valid_idx]
            mask[i, :valid_len] = 1

        return valid_value, mask
    
    

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
    