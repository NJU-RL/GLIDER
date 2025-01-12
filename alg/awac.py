from util.model import Policy, Critic
from alg.bc import Agent as BC_AGENT
from util.replay_buffer import SequenceDataset, batch_traj_process
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.utils.rnn import pad_sequence
import deepspeed
import torch
import copy


class ActorCritic(Policy):
    def __init__(self, args):
        super(ActorCritic, self).__init__(args)
        hidden_dim = self.base.config.hidden_size
        self.critic = Critic(hidden_dim)
        self.target_critic = Critic(hidden_dim)
        self.soft_update_target_critic(1.0)

    def soft_update_target_critic(self, tau):
        for target_param, param in zip(self.target_critic.parameters(), 
                                       self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) 
                                    + param.data * tau)
            
class Agent:
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
        
        if args['load_bc_model']:
            path = f"{args['check_path']}/{args['benchmark']}/behavior_clone/{args['model_name']}"
            BC_AGENT.load_policy(self, path)
        
        self.loss_fct = torch.nn.MSELoss()
        self.buffer = SequenceDataset(args)

        if self.engine.global_rank == 0:
            log_dir = f"{args['log_path']}/{args['benchmark']}/{args['alg_name']}/{args['model_name']}"
            self.writer = SummaryWriter(log_dir=log_dir)

        self.global_step = torch.tensor(0, dtype=torch.int64).to(self.engine.device)
        self.checkpoint_dir = f"{args['check_path']}/{args['benchmark']}/{args['alg_name']}/{args['model_name']}"

    def learn(self,):
        batch_size_per_gpu = self.engine.train_micro_batch_size_per_gpu()
        sampler = DistributedSampler(self.buffer, 
                                     num_replicas=self.engine.world_size, 
                                     rank=self.engine.local_rank)
        
        dataloader = DataLoader(self.buffer, 
                                batch_size=batch_size_per_gpu, 
                                sampler=sampler,
                                collate_fn=SequenceDataset.collate_fn)
        
        for epoch in range(self.args['epochs']):
            sampler.set_epoch(epoch)    # each epoch shuffle
            for batch in dataloader:
                """
                Args:
                    batch:{
                        "task_description": [traj_nums,],
                        "obs":[traj_nums, steps],
                        "action":[traj_nums, steps],
                        "next_obs":[traj_nums, steps]
                        "reward": [traj_nums, steps]
                        "done": [traj_nums, steps]
                    }
                Update Critic:
                    L_Q(\phi) = E_{s,a,r,s'~D}[Q_{\phi}(s,a) - r - \gamma * V_{\bar{\psi}(s')]
                    L_V(\psi) = E_{s~D}[E_{a~\pi_\theta(·|s)}[V_\psi(s)-Q_{\bar{\theta}}(s,a)]]
                Update Actor:
                    L_\pai(\theta) = -E_{s,a~D}[exp(1/lammda * A(s,a)) * log\pai_\theta(a|s)]
                """
                batch_tokens = batch_traj_process(batch['task_description'],
                                                  batch['obs'],
                                                  batch['action'],
                                                  self.engine.tokenizer).to(self.engine.device)
                rewards, dones = self.prepare_tensor(batch['score'], batch['done'])
                
                # Q(s,a), V(s)
                with torch.no_grad():
                    hidden_states, state_end_mask, action_end_mask = self.engine.get_hidden_states(batch_tokens)
                    target_vs, target_qsa = self.engine.target_critic(hidden_states)
                    target_vs, _ = self.extract_valid(target_vs, state_end_mask)
                    target_qsa, _ = self.extract_valid(target_qsa, action_end_mask)

                vs, q_sa = self.engine.critic(hidden_states)
                vs, _,  = self.extract_valid(vs, state_end_mask)
                q_sa, _ = self.extract_valid(q_sa, action_end_mask)

                # L_Q
                target = rewards[:, :-1] + (1-dones[:, :-1]) * target_vs[:, 1:] * self.args['gama']    # (batch, steps-1)
                q_loss = self.loss_fct(q_sa[:,:-1], target)

                # with torch.no_grad():
                #     target_v = self.get_policy_q(batch['task_description'],
                #                                  batch['obs'],
                #                                  batch['action'])
                v_loss = vs - target_qsa
                weight = torch.where(v_loss<0,
                                     torch.ones_like(vs)*(1-self.args['weight_tau']),
                                     torch.ones_like(vs)*self.args['weight_tau'])
                v_loss = (weight * (v_loss**2)).mean()
                self.engine.backward(q_loss+v_loss)
                self.engine.step()
                self.engine.soft_update_target_critic(tau=self.args['tau'])

                # log_prob 
                action_log_probs, action_masks = self.engine.get_log_prob(batch_tokens)
                # valid_action_log_probs,_ = self.extract_valid(action_log_probs, action_end_mask[:, 1:])
                valid_action_log_probs = BC_AGENT.extract_valid_action_probs(self, action_log_probs, action_masks, q_sa.size(1))
                if self.args['use_adv']:
                    # Advantage: Q(s,a)-V(s)
                    with torch.no_grad():
                        adv = q_sa - vs
                    actor_loss = -torch.mean(adv *valid_action_log_probs)
                else:
                    actor_loss = -torch.mean(q_sa.detach()*valid_action_log_probs)
                self.engine.backward(actor_loss)
                self.engine.step()

                if self.engine.local_rank == 0:  # Only log on the main process
                    self.writer.add_scalar('Loss/q_loss', q_loss.item(), self.global_step.item())
                    self.writer.add_scalar('Loss/v_loss', v_loss.item(), self.global_step.item())
                    self.writer.add_scalar('Loss/critic_loss', (q_loss+v_loss).item(), self.global_step.item())
                    self.writer.add_scalar('Loss/actor_loss', actor_loss.item(), self.global_step.item())
                    print(f"train; step:{self.global_step.item()}; actor_loss:{actor_loss.item()}; critic_loss:{q_loss.item(), v_loss.item(), (q_loss+v_loss).item()}")
                if self.global_step.item() % self.args['eval_freq'] == 0:
                    BC_AGENT.save_policy(self)
                self.global_step += 1
        BC_AGENT.save_policy(self)

    def get_policy_q(self, batch_prompt, batch_obs_list, batch_action_list):
        """
        Args:
            batch_obs_list: List[List[str]], shape: (batch, steps)
            batch_action_list: List[List[int]], shape: (batch, steps)
        Returns:
            q_values: Q(s, a~π), (batch, max_steps)
        """
        q_values = []
        for prompt, obs_list, action_list in zip(batch_prompt, batch_obs_list, batch_action_list):
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
        reward_list = [torch.tensor(seq, dtype=torch.float32, device=self.engine.device) for seq in rewards]
        done_list = [torch.tensor(seq, dtype=torch.int8, device=self.engine.device) for seq in dones]
        
        # padding
        reward_tensor = pad_sequence(reward_list, batch_first=True, padding_value=0.0)
        done_tensor = pad_sequence(done_list, batch_first=True, padding_value=0)
        
        return reward_tensor, done_tensor