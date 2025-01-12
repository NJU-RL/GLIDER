# import os
# import GPUtil

# def get_free_gpus(num_gpus_needed):
#     # 使用GPUtil获取空闲GPU
#     available_gpus = GPUtil.getAvailable(order='memory', limit=num_gpus_needed, maxLoad=0.1, maxMemory=0.1, includeNan=False, excludeID=[], excludeUUID=[])
#     return available_gpus

# num_gpus = int(os.environ.get('WORLD_SIZE', 1))

# free_gpus = get_free_gpus(num_gpus)
# os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, free_gpus))
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding
import torch
import deepspeed

model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.truncation_side = 'left'
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# llm = AutoModelForCausalLM.from_pretrained(model_name)
# engine, _, _, _ = deepspeed.initialize(model=llm,
#                                        model_parameters=[p for p in llm.parameters() if p.requires_grad],
#                                        config='./config/zero2.json'
#                                        )

with open("./prompt/icl_examples/scienceworld_icl.json", 'r') as f:
    example1 = json.load(f)
with open("./prompt/icl_examples/scienceworld_icl2.json", 'r') as f:
    example2 = json.load(f)

ex1_obs = example1["obs"]
ex2_obs = example2['obs']
ex1_action = example1["action"]
ex2_action = example2["action"]

example_obs = [ex1_obs, ex2_obs]
example_action = [ex1_action,ex2_action]

example_prompt = ["a", "b"]

from util.replay_buffer import batch_traj_process

# print(batch_traj_process(example_prompt, example_obs, example_action, tokenizer))



# def process_trajectory(prompts, predicts, tokenizer):
#     input_tensors = []
#     mask_tensors = []
#     labels = []
#     for prompt, predict in zip(prompts, predicts):
#         prompt_token = tokenizer(prompt, return_tensors='pt')
#         predict_token = tokenizer(predict, return_tensors='pt')

#         prompt_label = torch.full_like(prompt_token['input_ids'], -100)
#         predict_label = predict_token['input_ids']
        
#         input_tensors.extend([prompt_token['input_ids'], predict_token['input_ids']])
#         mask_tensors.extend([prompt_token['attention_mask'], predict_token['attention_mask']])
#         labels.extend([prompt_label, predict_label])
    
#     return BatchEncoding({
#         "input_ids": torch.cat(input_tensors, dim=1),
#         "attention_mask": torch.cat(mask_tensors, dim=1),
#         "labels": torch.cat(labels, dim=1)  
#     })


# traj_token = process_trajectory(ex1_obs, ex1_action, tokenizer).to(engine.device)
# output = engine(**traj_token)

# logits = output.logits[:, :-1, :]  # [batch_size, seq_len-1, vocab_size]
# labels = traj_token['labels'][:, 1:]  # [batch_size, seq_len-1]
# loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)  # 显式设置ignore_index
# loss = loss_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
# print(loss)



# from transformers import DataCollatorWithPadding

# def batch_traj_process(batch_prompts, batch_predicts, tokenizer):
#     batch_input, batch_mask, batch_labels = [], [], []
#     for prompts, predicts in zip(batch_prompts, batch_predicts):
#         input_tensors, mask_tensors, labels = [], [], []
#         for prompt, predict in zip(prompts, predicts):
#             prompt_token = tokenizer(prompt, return_tensors='pt')
#             predict_token = tokenizer(predict, return_tensors='pt')

#             prompt_label = torch.full_like(prompt_token['input_ids'], -100)
#             predict_label = predict_token['input_ids']
            
#             input_tensors.extend([prompt_token['input_ids'], predict_token['input_ids']])
#             mask_tensors.extend([prompt_token['attention_mask'], predict_token['attention_mask']])
#             labels.extend([prompt_label, predict_label])
#         batch_input.append(torch.cat(input_tensors, dim=1))
#         batch_mask.append(torch.cat(mask_tensors, dim=1))
#         batch_labels.append(torch.cat(labels, dim=1))

#     # Padding到最长的batch长度
#     padded_input = torch.nn.utils.rnn.pad_sequence([x.squeeze(0) for x in batch_input], 
#                                                   batch_first=True, 
#                                                   padding_value=tokenizer.pad_token_id)
#     padded_mask = torch.nn.utils.rnn.pad_sequence([x.squeeze(0) for x in batch_mask], 
#                                                  batch_first=True, 
#                                                  padding_value=0)
#     padded_labels = torch.nn.utils.rnn.pad_sequence([x.squeeze(0) for x in batch_labels], 
#                                                    batch_first=True, 
#                                                    padding_value=-100)
    
#     return BatchEncoding({
#         "input_ids": padded_input,
#         "attention_mask": padded_mask,
#         "labels": padded_labels
#     })



# traj_token = batch_traj_process([ex1_obs,ex2_obs], [ex1_action,ex2_action], tokenizer)
# with open('aaaa.json', 'w') as json_file:
#     json.dump({traj_token["input_ids"]}, json_file, indent=4)

# output = engine(**traj_token)
# logits = output.logits[:, :-1, :]  # [batch_size, seq_len-1, vocab_size]
# labels = traj_token['labels'][:, 1:]  # [batch_size, seq_len-1]
# loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)  # 显式设置ignore_index
# loss = loss_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
# print(loss)
