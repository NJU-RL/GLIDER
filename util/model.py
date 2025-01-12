import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model

class Policy(nn.Module):
    def __init__(self, args):
        super(Policy, self).__init__()
        self.args = args

        self.tokenizer = AutoTokenizer.from_pretrained(args["model_name"])
        self.tokenizer.truncation_side = 'left'
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.base = AutoModelForCausalLM.from_pretrained(args["model_name"])
        if args["use_lora"]:
            lora_config = LoraConfig(
                r=16,
                target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                task_type=TaskType.CAUSAL_LM,
                lora_alpha=32,
                lora_dropout=0.05
            )
            self.base = get_peft_model(self.base, lora_config)
        # self.hidden_dim = self.base.config.hidden_size

    def generate_action(self, state_ids):
        state_ids = state_ids.to(self.base.device)  # (batch, sqe_len)
        context_len = state_ids['input_ids'].size(1)
        outputs = self.base.generate(**state_ids, 
                                     max_new_tokens=self.args["max_new_tokens"],
                                    #  do_sample=self.args["do_sample"], 
                                    #  temperature=self.args["temperature"],
                                     pad_token_id=self.tokenizer.eos_token_id
                                     )
        
        raw_action = self.tokenizer.batch_decode(outputs[:, context_len:],
                                                 skip_special_tokens=True)


        return raw_action

    def get_log_prob(self, traj_token):
        """
        Args:
            traj_token: {
                "input_ids": [batch_size, seq_len],
                "attention_mask": [batch_size, seq_len],
                "labels": [batch_size, seq_len] (-100 is not action token)
                }
        formate -> (prompt, s1, a1, s2, a2, ..., st, at, st+1)
        Returns: 
            action_log_probs: [batch_size, seq_len-1]
        """
        # output = self.base(**traj_token)
        output = self.base(input_ids=traj_token['input_ids'], 
                           attention_mask=traj_token['attention_mask'])
        logits = output.logits[:, :-1, :]   # (batch, seq_len-1, vacab_size)
        labels = traj_token['labels'][:, 1:]    # (batch, seq_len-1)
        action_masks = (labels != -100).float() # (batch, seq_len-1)
        log_probs = torch.log_softmax(logits, dim=-1)   # (batch, seq_len-1, vacab_size)
        # gather action log_prob
        action_log_probs = torch.gather(log_probs, 2,
                                        labels.unsqueeze(-1).clamp(min=0) # clip -100 to 0
                                        ).squeeze(-1) # (batch, seq_len-1)
        
        return action_log_probs, action_masks
    

    def get_hidden_states(self, traj_token):
        """
        traj_token struct: prompt -> (state -> action)* -> padding
        Returns:
            hidden_states:(batch, seq_len, hidden_dim)
            state_end_mask: (batch, seq_len) (state end token set to 1, other is 0)
            action_end_mask: (batch, seq_len) (action end token set to 1, other is 0)
        """
        # outputs = self.base(**traj_token, output_hidden_states=True)
        outputs = self.base(input_ids=traj_token['input_ids'], 
                           attention_mask=traj_token['attention_mask'],
                           output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1] #(batch, seq_len, hidden_dim)
        
        return hidden_states, traj_token["state_end_mask"], traj_token["action_end_mask"]


class Critic(nn.Module):
    def __init__(self, hidden_dim):
        super(Critic, self).__init__()
        # V(s)
        self.value = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, 1))
        # Q(s,a)
        self.q_value = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, 1))
        
    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
        Returns:
            values: (batch, seq_len) -> valid: (batch, num_states)
            q_values: (batch, seq_len) -> valid: (batch, num_state_actions)
        """
        values = self.value(hidden_states).squeeze(-1)
        q_values = self.q_value(hidden_states).squeeze(-1)

        return values, q_values
