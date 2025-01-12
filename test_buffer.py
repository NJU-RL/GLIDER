# from util.replay_buffer import SequenceDataset, batch_traj_process
# import json

# with open("./config/default.json", 'r') as f:
#     args = json.load(f)
# buffer = SequenceDataset(args)

# buffer.load_data()
# with open('train.json', 'w') as json_file:
#     json.dump(buffer.data, json_file, indent=4)

# print(buffer.__len__())

# from transformers import AutoModelForCausalLM, AutoTokenizer

# model_name = "meta-llama/Meta-Llama-3-8B"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.truncation_side = 'left'
# tokenizer.padding_side = "left"
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.pad_token_id = tokenizer.eos_token_id

# batch_tokens = batch_traj_process(buffer.data['task_description'],buffer.data['obs'], buffer.data['action'], tokenizer)
# print(batch_tokens["input_ids"].size())

# import torch

# labels = torch.tensor([[-100, -100, 1, 324, 9, 0, -100, -100, 2, 46, 0, 5],
#                        [-100, -100, 3, 4, 5, 0, 2, -100, -100, -100, -100, -100]])

# action_start =(labels != -100).nonzero(as_tuple=True)
# print((labels != -100))
# print(action_start)

from util.replay_buffer import HierarchyDataset
import json
with open("./config/glider_bc.json", 'r') as f:
    args = json.load(f)
buffer = HierarchyDataset(args)
buffer.convert_data()