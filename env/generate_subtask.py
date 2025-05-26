import os
import GPUtil

def get_free_gpus(num_gpus_needed):
    # 使用GPUtil获取空闲GPU
    available_gpus = GPUtil.getAvailable(order='memory', limit=num_gpus_needed, maxLoad=0.1, maxMemory=0.1, includeNan=False, excludeID=[], excludeUUID=[])
    return available_gpus

num_gpus = int(os.environ.get('WORLD_SIZE', 1))


free_gpus = get_free_gpus(num_gpus)
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, free_gpus))

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import deepspeed
import json

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
# llm = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
llm = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")  
engine, _, _, _ = deepspeed.initialize(model=llm,
                                       model_parameters=[p for p in llm.parameters() if p.requires_grad],
                                       config='./config/zero2.json'
                                       )


example1_prompt = {
    "task_description": "Task Description:\nYour task is to find a(n) animal. First, focus on the thing. Then, move it to the red box in the kitchen.",
    "subtask": [],
    "action": [
        "open door to greenhouse",
        "go to greenhouse",
        "open door to outside",
        "go to outside",
        "look around",
        "focus on butterfly",
        "pick up butterfly",
        "open door to kitchen",
        "go to kitchen",
        "move egg butterfly egg in inventory to red box"
    ]
}
example_1= {
            "subtask": [
                "Find an animal and focus it",
                "Navigation to kitchen with the animal",
                "Move the animal to the red box in the kitchen"
            ],
            "action": [
                ["open door to greenhouse",
                "go to greenhouse",
                "open door to outside",
                "go to outside",
                "look around",
                "focus on butterfly"],
                
                ["pick up butterfly",
                 "open door to kitchen",
                 "go to kitchen"],

                ["move egg butterfly egg in inventory to red box"]
                ]
}

example2_prompt = {
    "task_description": "Task Description:\nYour task is to find a(n) animal. First, focus on the thing. Then, move it to the purple box in the workshop.",
    "subtask": [],
    "action": [
        "open door to hallway",
        "go to hallway",
        "open door to kitchen",
        "go to kitchen",
        "open door to outside",
        "go to outside",
        "look around",
        "focus on blue jay",
        "pick up blue jay",
        "open door to kitchen",
        "go to kitchen",
        "open door to hallway",
        "go to hallway",
        "open door to workshop",
        "go to workshop",
        "move egg blue jay egg in inventory to purple box"
    ]
}

example_2 = {
    "task_description": "Task Description:\nYour task is to find a(n) animal. First, focus on the thing. Then, move it to the purple box in the workshop.",
    "subtask": [
        "Find an animal and focus it"
        "Navigation to webshop with the animal",
        "Move the animal to the purple box in the webshop"
    ],
    "action": [
        ["open door to hallway",
        "go to hallway",
        "open door to kitchen",
        "go to kitchen",
        "open door to outside",
        "go to outside",
        "look around",
        "focus on blue jay"],

        ["pick up blue jay",
        "open door to kitchen",
        "go to kitchen",
        "open door to hallway",
        "go to hallway",
        "open door to workshop",
        "go to workshop"],
        
        ["move egg blue jay egg in inventory to purple box"]
    ]
}

target_prompt = {
    "task_description": "Task Description:\nYour task is to find a(n) animal. First, focus on the thing. Then, move it to the purple box in the bathroom.",
    "subtask": [],
    "action": [
        "open door to greenhouse",
        "go to greenhouse",
        "open door to outside",
        "go to outside",
        "look around",
        "focus on egg frog",
        "pick up egg frog",
        "open door to kitchen",
        "go to kitchen",
        "open door to bathroom",
        "go to bathroom",
        "move living thing in inventory to purple box"
    ]
}

def generate_response(prompt, engine, tokenizer):
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device=engine.local_rank)
    context_len = inputs['attention_mask'].size(1)
    print(f"attention_mask.shape: {inputs['attention_mask'].shape}")
    # Generate
    with torch.no_grad():
        outputs = engine.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True
        )
    
    # Decode response
    response = tokenizer.batch_decode(outputs[:,context_len:], skip_special_tokens=True)
    return response


def create_prompting_template(example1_prompt, example_1, example2_prompt, example_2, target_prompt):
    prompt = f"""Given a task with an actions list, I want you to organize it into subtasks with corresponding grouped actions. Here are two examples:
    Example 1:
    Task Description: {example1_prompt["task_description"]}
    Actions: {example1_prompt["action"]}
    Organized into:
    Subtasks: {example_1["subtask"]}
    Grouped Actions: {example_1["action"]}

    Example 2:
    Task Description: {example2_prompt["task_description"]}
    Actions: {example2_prompt["action"]}
    Organized into:
    Subtasks: {example_2["subtask"]}
    Grouped Actions: {example_2["action"]}
    
    Now, please organize the following task in the same way:
    Task Description: {target_prompt["task_description"]}
    Actions: {target_prompt["action"]}
    Please organize this into subtasks and group the actions, directly:"""
    return prompt





task_id = 5
var_nums = 120
for i in range(var_nums):
    # target:
    with open(f'dataset/scienceworld/task{task_id}/variation{3}.json', 'r') as json_file:
        target_prompt = json.load(json_file)

    # Example response structure expected from the LLM


    # Output the prompt
    prompt = create_prompting_template(example1_prompt, example_1, example2_prompt, example_2, target_prompt)
    # print(prompt)
    # Generate response
    response = generate_response(prompt, engine, tokenizer)

    # Print results
    print("\nGenerated Response:")
    print(response)
    
    exit()



# # Prompt to LLM
# def create_prompting_template(example_prompt, example, target_prompt):
#     prompt = f"""Given a task with actions, I want you to organize it into subtasks with corresponding grouped actions. Here's an example:
#     Task Description: {example_prompt["task_description"]}
#     Actions: {example_prompt["action"]}
#     Organized into:
#     Subtasks: {example["subtask"]}
#     Grouped Actions: {example["action"]}
#     As you can see, the actions are grouped under relevant subtasks. Each subtask has its corresponding list of actions that accomplish that subtask.
#     Now, please organize the following task in the same way:
#     Task Description: {target_prompt["task_description"]}
#     Actions: {target_prompt["action"]}
#     Please organize this into subtasks and group the actions accordingly. Structure your response as a Python dictionary with "subtask" and "action" as keys, where "action" contains nested lists corresponding to each subtask."""
    
#     return prompt
    
#  def generate_response(prompt, engine, tokenizer, max_length=1024, temperature=0.1):
#     # Encode the prompt
#     inputs = tokenizer(prompt, return_tensors="pt").to(engine.local_rank)
#     context_len = inputs['attention_mask'].size(1)
#     print(f"attention_mask.shape: {inputs['attention_mask'].shape}")
#     # Generate response
#     with torch.no_grad():
#         outputs = engine.generate(
#             **inputs,
#             max_length=max_length,
#             # temperature=temperature,
#             num_return_sequences=1,
#             pad_token_id=tokenizer.eos_token_id,
#             do_sample=False
#         )
    
#     # Decode and return the response
#     response = tokenizer.batch_decode(outputs[:,context_len:], skip_special_tokens=True)
#     return response