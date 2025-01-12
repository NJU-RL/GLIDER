import os
import deepspeed
import torch
from transformers import AutoTokenizer
from util.model import Policy
from util.replay_buffer import HierarchyDataset, batch_traj_process
from alg.bc import Agent
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from prompt.inst import high_prompt, low_prompt

class GLIDER:
    def __init__(self, args):
        self.args = args
        hierarcy_policy = Policy(args)
        self.engine, _ , _, _ = deepspeed.initialize(model=hierarcy_policy,
                                                     model_parameters=[{"params": [p for p in hierarcy_policy.base.parameters() if p.requires_grad], 
                                                                        "lr": args["lr"]}],
                                                     config=args["ds_config"])
        
        self.buffer = HierarchyDataset(args)

        if self.engine.global_rank == 0:
            log_dir = f"{args['log_path']}/{args['benchmark']}/{args['alg_name']}/{args['model_name']}"
            self.writer = SummaryWriter(log_dir=log_dir)

        self.global_step = torch.tensor(0, dtype=torch.int64).to(self.engine.device)
        self.checkpoint_dir = f"{args['check_path']}/{args['benchmark']}/{args['alg_name']}/{args['model_name']}"       

    def update_policy(self):
        batch_size_per_gpu = self.engine.train_micro_batch_size_per_gpu()
        sampler = DistributedSampler(self.buffer, 
                                     num_replicas=self.engine.world_size, 
                                     rank=self.engine.local_rank)
        dataloader = DataLoader(self.buffer, 
                                batch_size=batch_size_per_gpu, 
                                sampler=sampler,
                                collate_fn=HierarchyDataset.collate_fn)
        for epoch in range(self.args['epochs']):
            high_epoch_loss, low_epoch_loss = 0.0, 0.0
            sampler.set_epoch(epoch)    # each epoch shuffle
            for batch in dataloader:
                """
                Args:
                    batch:{
                        "task_description": [traj_nums,] or "subtask":[group_nums, ]
                        "obs":[traj_nums, steps+1],      or "obs": [group_nums, steps+1]
                        "subtask":[traj_nums, steps],    or "action": [group_nums, steps]
                        ...
                        }
                    pi(a|s)/action_token_len   
                """
                # high level
                batch_high_tokens = batch_traj_process(batch['high']['task_description'],
                                                       batch['high']['obs'],
                                                       batch['high']['subtask'],
                                                       self.engine.tokenizer).to(self.engine.device)

                high_log_probs, high_masks = self.engine.get_log_prob(batch_high_tokens)
                high_valid_log_prob = Agent.extract_valid_action_probs(self, high_log_probs, high_masks,
                                                                       max(batch_high_tokens['action_end_mask'].sum(dim=1)))
                high_loss = -high_valid_log_prob.mean()

                # low level
                batch_low_tokens = batch_traj_process(batch['low']['subtask'],
                                                      batch['low']['obs'],
                                                      batch['low']['action'],
                                                      self.engine.tokenizer).to(self.engine.device)
                low_log_probs, low_masks = self.engine.get_log_prob(batch_low_tokens)
                low_valid_log_prob = Agent.extract_valid_action_probs(self, low_log_probs, low_masks,
                                                                       max(batch_low_tokens['action_end_mask'].sum(dim=1)))
                low_loss = -low_valid_log_prob.mean()

                self.engine.backward(high_loss+low_loss)
                self.engine.step()

                high_epoch_loss += high_loss.item()
                low_epoch_loss += low_loss.item()
                self.global_step += 1

                if self.engine.local_rank == 0:  # Only log on the main process
                    print(f"train; step:{self.global_step.item()}; high:{high_loss.item()}; low:{low_loss.item()}; loss:{(low_loss+high_loss).item()}")
                    self.writer.add_scalar('step_loss/high', high_loss.item(), self.global_step.item())
                    self.writer.add_scalar('step_loss/low', low_loss.item(), self.global_step.item())
                    self.writer.add_scalar('step_loss/bc', (high_loss+low_loss).item(), self.global_step.item())
            
            if self.engine.local_rank == 0: # Only log on the main process
                print(f"hierarcy-train; epoch{epoch}, high:{high_epoch_loss}, low:{low_epoch_loss}")
                Agent.save_policy(self)
                self.writer.add_scalar('epoch_loss/high', high_epoch_loss, epoch)
                self.writer.add_scalar('epoch_loss/low', low_epoch_loss, epoch)
                self.writer.add_scalar('epoch_loss/bc', high_epoch_loss+low_epoch_loss, epoch)
            