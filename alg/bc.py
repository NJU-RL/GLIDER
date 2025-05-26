import os
import deepspeed
import torch
from transformers import AutoTokenizer
from util.model import Policy
from util.replay_buffer import SequenceDataset, batch_traj_process
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

class Agent:
    def __init__(self, args):
        self.args = args
        policy = Policy(args) 
        self.engine, _ , _, _ = deepspeed.initialize(model=policy,
                                                    model_parameters=[p for p in policy.parameters() if p.requires_grad],
                                                    config=args["ds_config"])
        
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
                        ...
                    }
                pi(a|s)/action_token_len    
                """
                batch_tokens = batch_traj_process(batch['task_description'],
                                                  batch['obs'],
                                                  batch['action'],
                                                  self.engine.tokenizer).to(self.engine.device)
                
                action_log_probs, action_masks = self.engine.get_log_prob(batch_tokens)  # (batch, seq_len-1)
                action_num = batch_tokens["action_end_mask"].sum(dim=1)
                valid_log_prob = self.extract_valid_action_probs(action_log_probs, action_masks, max(action_num))
                bc_loss = -valid_log_prob.mean()
                # bc_loss = -(torch.sum(action_log_probs*action_masks, dim=1)/torch.sum(action_masks, dim=1)).mean()
                self.engine.backward(bc_loss)
                self.engine.step()

                if self.engine.local_rank == 0:  # Only log on the main process
                    self.writer.add_scalar('Loss/bc_loss', bc_loss.item(), self.global_step.item())
                    print(f"train; step:{self.global_step.item()}; Loss:{bc_loss}")
                if self.global_step.item() % self.args['eval_freq'] == 0:
                    self.save_policy()
                self.global_step += 1  
        self.save_policy()  

    def extract_valid_action_probs(self, log_probs, masks, max_action_nums):
        """
        Args:
            log_probs: (batch, seq_len-1)
            masks: (batch, seq_len-1), 1 is action token position
            max_action_nums: int, maximum number of actions
        Returns:
            valid_action_probs: (batch, max_action_nums)
        """
        batch_size = log_probs.size(0)
        valid_action_probs = torch.zeros(batch_size, max_action_nums, device=log_probs.device)
        
        for i in range(batch_size):
            action_positions = torch.where(masks[i]==1)[0]
            
            action_groups = []
            current_group = []
            for pos in action_positions:
                if not current_group or pos==current_group[-1]+1:
                    current_group.append(pos)
                else:
                    action_groups.append(current_group)
                    current_group = [pos]
            if current_group:
                action_groups.append(current_group)

            # average group log_prob
            for j, group in enumerate(action_groups):
                group_probs = log_probs[i, group]
                # valid_action_probs[i, j] = group_probs.sum()
                valid_action_probs[i, j] = group_probs.sum() / len(group)

        return valid_action_probs

    def save_policy(self):
        if self.engine.local_rank == 0:
            if self.args["use_lora"]:
                #  Save LoRA weights directly using PEFT's save_pretrained method
                self.engine.module.base.save_pretrained(self.checkpoint_dir)
                self.engine.module.tokenizer.save_pretrained(self.checkpoint_dir)
            else:
                # Save normal model weights
                model_path = os.path.join(self.checkpoint_dir, "policy.pth")
                torch.save(self.engine.module.base.state_dict(), model_path)
                self.engine.module.tokenizer.save_pretrained(self.checkpoint_dir)

    def load_policy(self, path):
        if os.path.exists(path):
            if self.args["use_lora"]:
                # load LoRA weights
                self.engine.module.base.load_adapter(path, adapter_name="default")
            else:
                # Load normal model weights
                model_path = os.path.join(path, "policy.pth")
                if os.path.exists(model_path):
                    self.engine.module.base.load_state_dict(torch.load(model_path))
            
            # Load tokenizer
            self.engine.module.tokenizer = AutoTokenizer.from_pretrained(path)
            print(f"Model loaded from {path}")
        else:
            print(f"No checkpoint found at {path}")

