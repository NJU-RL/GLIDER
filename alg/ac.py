from util.model import ActorCritic
import deepspeed
from torch.distributions import Categorical
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import torch
import torch.nn as nn
from scienceworld import ScienceWorldEnv
import numpy as np
from util.replay_buffer import OfflineBuffer
from torch.utils.tensorboard import SummaryWriter



class Agent:
    def __init__(self, args):
        self.args = args

        actor_critic = ActorCritic(args)
        self.engine, _ , _, _ = deepspeed.initialize(model=actor_critic,
                                                     model_parameters=[{"params": [p for p in actor_critic.actor.parameters() if p.requires_grad], 
                                                                        "lr": args["actor_lr"]},
                                                                       {"params": [p for p in actor_critic.critic.parameters() if p.requires_grad], 
                                                                        "lr": args["critic_lr"]}],
                                                     config=args["ds_config"])
                                                    
                                                                 
        self.MseLoss = nn.MSELoss()
        self.offline_buffer = OfflineBuffer(args["task_id"])

        if self.engine.global_rank == 0:  # Only log on the main process
            log_dir = f"./logs/offline_ac/task{args['task_id']}/rank_{self.engine.local_rank}"
            self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = torch.tensor(0, dtype=torch.int64).to(self.engine.device)

        self.eval_env = ScienceWorldEnv("", envStepLimit=args['env_step_limit'])
        self.task_name = self.eval_env.getTaskNames()[args["task_id"]]
        self.max_variations = self.eval_env.getMaxVariations(self.task_name)
        

    def get_action(self, obs):
        obs_ids = self.str2token(obs).to(self.engine.local_rank) # (batch, context_len, dim)
        action_prob, obj_prob1, obj_prob2 = self.engine(obs_ids, "actor")
        action_dist, obj1_dist, obj2_dist = Categorical(probs=action_prob), Categorical(probs=obj_prob1), Categorical(probs=obj_prob2)
        action, obj1, obj2 = action_dist.sample(), obj1_dist.sample(), obj2_dist.sample()
        a_logprob, obj1_logprob, obj2_logprob = action_dist.log_prob(action), obj1_dist.log_prob(obj1), obj2_dist.log_prob(obj2) #(batch_size, 1)
        action_obj_concat = self.engine.action_space.get_action_obj_combination(action, obj1, obj2)
        return action_obj_concat[0], action.flatten(), obj1.flatten(), obj2.flatten(), (a_logprob+obj1_logprob+obj2_logprob).flatten()


    def str2token(self, str):
        return self.engine.tokenizer(str, padding=True, truncation=True, return_tensors="pt")
        # return self.tokenizer(str,
        #                     add_special_tokens=True,
        #                     return_token_type_ids=False,
        #                     padding=True,
        #                     return_attention_mask=True,
        #                     return_tensors='pt',
        #                     truncation=True,
        #                     max_length=1024)
    
    def evaluate(self, obs):  # When evaluating the policy, we select the action with the highest probability
        obs_ids = self.str2token(obs).to(self.engine.local_rank)
        action_prob, obj_prob1, obj_prob2 = self.engine(obs_ids, "actor")
        action, obj1, obj2 = torch.argmax(action_prob, dim=1), torch.argmax(obj_prob1, dim=1), torch.argmax(obj_prob2, dim=1)
        action_obj_concat = self.engine.action_space.get_action_obj_combination(action, obj1, obj2)
        return action_obj_concat[0]
    
    def learn(self):
        batch_size_per_gpu = self.engine.train_micro_batch_size_per_gpu()
        sampler = DistributedSampler(self.offline_buffer, num_replicas=self.engine.world_size, rank=self.engine.local_rank)
        dataloader = DataLoader(self.offline_buffer, batch_size=batch_size_per_gpu, sampler=sampler)
        device = self.engine.local_rank

        for epoch in range(self.args['epochs']):
            sampler.set_epoch(epoch)    # each epoch shuffle
            for batch in dataloader:
                obs_ids, obs_ids_ = self.str2token(batch["obs"]).to(device), self.str2token(batch["next_obs"]).to(device)
                vs = self.engine(obs_ids, "critic").flatten()  #(batch,)
                vs_ = self.engine(obs_ids_, "critic").flatten()

                with torch.no_grad():
                    td_target = batch["score"].to(device) + self.args["gama"] * (1 - batch["done"].to(device)) * vs_

                action_prob, obj_prob1, obj_prob2=self.engine(obs_ids, "actor")
                action_dist_now, obj1_dist_now, obj2_dist_now = Categorical(probs=action_prob), Categorical(probs=obj_prob1), Categorical(probs=obj_prob2)
                a_logprob = action_dist_now.log_prob(batch["action_id"].to(device))\
                            + obj1_dist_now.log_prob(batch["obj1_id"].to(device))\
                            + obj2_dist_now.log_prob(batch["obj2_id"].to(device))
                actor_loss = -((td_target - vs).detach()) * a_logprob
                self.engine.backward(actor_loss.mean())
                self.engine.step()

            
                critic_loss = self.MseLoss(td_target, vs)
                self.engine.backward(critic_loss)
                self.engine.step()

                # Synchronize steps across GPUs
                local_step = torch.tensor(1, dtype=torch.int64).to(device)
                dist.all_reduce(local_step, op=dist.ReduceOp.SUM)
                if self.engine.global_rank == 0:  # Only log on the main process
                    self.writer.add_scalar('Loss/actor_loss', actor_loss.mean().item(), self.global_step.item())
                    self.writer.add_scalar('Loss/critic_loss', critic_loss.item(), self.global_step.item())

                if self.global_step.item() % self.args['eval_freq'] == 0:
                    eval_variation_id = np.random.randint(self.offline_buffer.train_vari_nums, self.max_variations+1)
                    eval_reward, eval_score = self.eval_policy(eval_variation_id)
                    train_variation_id = np.random.randint(0,self.offline_buffer.train_vari_nums)
                    train_reward, train_score = self.eval_policy(train_variation_id)
                    if self.engine.local_rank == 0:
                        print(f"step:{self.global_step.item()}; Reward:{eval_reward}; Score: {eval_score}")
                        self.writer.add_scalar('eval/reward', eval_reward, self.global_step.item())
                        self.writer.add_scalar('eval/score', eval_score, self.global_step.item())
                        self.writer.add_scalar('train/reward', train_reward, self.global_step.item())
                        self.writer.add_scalar('train/score', train_score, self.global_step.item())
                # Update global step
                self.global_step += local_step

    def eval_policy(self, variation_id):
        
        eval_reward = 0.0
        eval_score = 0.0

        # variation_id = np.random.randint(self.offline_buffer.train_vari_nums, self.max_variations+1)
        self.eval_env.load(self.task_name, variation_id)
        obs, _ = self.eval_env.reset()
        done = False
        episode_steps = 0
        while not done:
            episode_steps += 1
            co_action_obj = self.evaluate([obs])
            obs_, reward, done, info = self.eval_env.step(co_action_obj)
            if episode_steps == self.args['env_step_limit']:
                done = True
            eval_reward += reward
            eval_score += info['score']
            obs = obs_

        eval_reward = torch.tensor(eval_reward).to(self.engine.device)
        dist.all_reduce(eval_reward, op=dist.ReduceOp.SUM)
        average_eval_reward = eval_reward.item() / self.engine.world_size

        eval_score = torch.tensor(eval_score).to(self.engine.device)
        dist.all_reduce(eval_score, op=dist.ReduceOp.SUM)
        average_eval_score = eval_score.item() / self.engine.world_size

        return average_eval_reward, average_eval_score

        # if self.engine.local_rank == 0:
        #     print(f"step:{self.global_step}; Reward:{average_eval_reward}; Score: {average_eval_score}")
            

