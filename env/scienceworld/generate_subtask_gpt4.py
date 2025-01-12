import openai
 
openai.api_key = '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'

def llm(prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    response = openai.ChatCompletion.create(
      model="gpt-4o",
      messages = messages,
    #   prompt=prompt,
      temperature=1,
      max_tokens=1024,
      top_p=1,
      frequency_penalty=0.0,
      presence_penalty=0.0,
    #   stop=stop
    )
    return response["choices"][0]["message"]["content"]

def apichatgpt_givenprompt(prompt, lm="gpt-4o"):
    """This function uses API instead of a language model.
    Here we use GPT3 API text-davinci-003 which is the most powerful model"""
    openai.api_key = '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'

    completions = openai.ChatCompletion.create(
        model=lm,
        n=1,
        messages=prompt,
        temperature=0.0,
    )

    instructions = completions.choices[0]["message"]["content"]

    return instructions

desc = (
    "You are in a simulated environment as an agent. "
    "A task and its description will be given to you. "
    "Suggest the best actions the agent can take based "
    "on the things you see and the items in your inventory to complete the task. "
    "Only use valid actions and objects. "
    "If you know what are around, then suggest "
    "the following actions. You are allowed to do the following actions with the objects. "
    "Open or close OBJ meaning open or close a container , Deactivate or activate OBJ meaning "
    "activate or deactivate a device, connect OBJ to OBJ meaning connect electrical components , "
    "disconnect OBJ meaning disconnect electrical components , use OBJ [on OBJ] meaning use a device/item ,"
    " look around meaning describe the current room, look at OBJ meaning describe an object in detail, "
    "look in OBJ meaning describe a container’s contents, read OBJ meaning read a note or book, "
    "move OBJ to OBJ meaning move an object to a container, pick up OBJ meaning move an object to the inventory,"
    " put down OBJ meaning drop an inventory item, pour OBJ into OBJ meaning pour a liquid into a container , "
    "dunk OBJ into OBJ meaning dunk a container into a liquid , mix OBJ meaning chemically mix a container , "
    "go to LOC meaning move to a new location , teleport to LOC meaning teleport to a specific room , "
    "eat OBJ meaning eat a food , flush OBJ meaning flush a toilet, focus on OBJ meaning signal intent "
    "on a task object, wait [DURATION] meaning take no action for some duration, task meaning describe "
    "current task, inventory meaning list agent’s inventory, OBJ means objects. LOC means location. "
    "There are 10 locations centered around a house theme. These are: kitchen, bathroom, workshop,  \
                     art studio, greenhouse, outside, bedroom, living room, foundry."
)

def create_initialprompt_start(sample1):
    """This function create samples for the prompting using the samples."""
    task_desc1 = "Your task is to boil water. For compounds without a boiling point, combusting the substance is also acceptable. First, focus on the substance. Then, take actions that will cause it to change its state of matter."
    path1 = [
        "open door to kitchen",
        "go to kitchen",
        "pick up thermometer",
        "open cupboard",
        "pick up metal pot",
        "move metal pot to sink",
        "activate sink",
        "deactivate sink",
        "pick up metal pot",
        "focus on substance in metal pot",
        "pour metal pot into metal pot",
        "pick up metal pot",
        "move metal pot to stove",
        "activate stove",
        "examine substance in metal pot",
        "use thermometer in inventory on substance in metal pot",
        "examine substance in metal pot",
        "use thermometer in inventory on substance in metal pot",
        "examine substance in metal pot",
        "use thermometer in inventory on substance in metal pot",
        "examine substance in metal pot",
        "use thermometer in inventory on substance in metal pot",
        "examine substance in metal pot",
        "use thermometer in inventory on substance in metal pot",
        "examine substance in metal pot",
        "use thermometer in inventory on substance in metal pot",
        "examine substance in metal pot",
        "use thermometer in inventory on substance in metal pot",
        "examine substance in metal pot",
        "use thermometer in inventory on substance in metal pot",
        "examine substance in metal pot",
        "use thermometer in inventory on substance in metal pot",
        "examine substance in metal pot",
        "use thermometer in inventory on substance in metal pot",
        "examine steam",
        "use thermometer in inventory on steam",
        "wait1",
        "wait1",
    ]
    goalpath1 = ""
    for step in path1:
        goalpath1 += step + ", "

    messages = [
        {"role": "system", "content": "You are a helpful assistant." + desc},
        {
            "role": "user",
            "content": task_desc1
            + "Here is the goal path to achieve to the goal:"
            + goalpath1
            + ". Based on the given goal path, provide me with the functional format of high-level sub-tasks to complete this task and their correspondings actions.",
        },
    ]

    messages.append(
        {
            "role": "assistant",
            "content": "1- navigate_to(kitchen) : {'open door to kitchen', 'go to kitchen'} 2- pick_up(thermometer):{'pick up thermometer'} 3- find(metal pot):{'open cupboard', 'pick up metal pot'} 4- Fill(metal pot, water): {'move metal pot to sink', 'activate sink', 'deactivate sink', 'pick up metal pot'} 5- Focus_on(substance in metal pot):{'focus on substance in metal pot'} 6- heat(water, metal pot): {'pour metal pot into metal pot', 'pick up metal pot', 'move metal pot to stove', 'activate stove'} 7- Monitor_temperature(metal pot, boil): {'examine substance in metal pot', 'use thermometer in inventory on substance in metal pot', 'examine substance in metal pot', 'use thermometer in inventory on substance in metal pot', 'examine substance in metal pot', 'use thermometer in inventory on substance in metal pot', 'examine substance in metal pot', 'use thermometer in inventory on substance in metal pot', 'examine substance in metal pot', 'use thermometer in inventory on substance in metal pot', 'examine substance in metal pot', 'use thermometer in inventory on substance in metal pot', 'examine substance in metal pot', 'use thermometer in inventory on substance in metal pot', 'examine substance in metal pot', 'use thermometer in inventory on substance in metal pot', 'examine substance in metal pot', 'use thermometer in inventory on substance in metal pot'} 8- chek(steam): {'examine steam', 'use thermometer in inventory on steam'} 9- wait(2): {'wait1', 'wait1'}",
        }
    )

    path_sample = sample1["gold_path"]
    goalpath_sample = ""
    for step in path_sample:
        goalpath_sample += step + ", "

    messages.append(
        {
            "role": "user",
            "content": "New task Description: "
            + sample1["task_desc"]
            + " Here is the goal path to achieve to the goal: "
            + goalpath_sample
            + ". Based on the given goal path, provide me with the functional format of high-level sub-tasks to complete this task and their correspondings actions. ",
        }
    )

    ## Call GPT4 or ChatGPT
    GPT4output = apichatgpt_givenprompt(messages)
    messages.append({"role": "assistant", "content": GPT4output})

    return GPT4output



from scienceworld import ScienceWorldEnv
import json
import os

TASK_NUM = 30

env = ScienceWorldEnv("", envStepLimit=1000)
task_names = env.getTaskNames()
data_type = "train"

for task_id in range(TASK_NUM):
    env.load(task_names[task_id], generateGoldPath=True)
    if data_type == "train":
        vari_ids = env.getVariationsTrain()
    elif data_type == "dev":
        vari_ids = env.getVariationsDev()
    elif data_type == "test":
        vari_ids = env.getVariationsTest()
    save_path = f"datasets/subtask_data/task{task_id}/{data_type}/"
    os.makedirs(save_path, exist_ok=True)
    for vari in vari_ids:
        with open(f'datasets/raw_data/task{task_id}/{data_type}/variation{vari}.json', 'r') as json_file:
            raw_data = json.load(json_file)

        sample = {
            "task_desc": raw_data['task_description'],
            "gold_path": raw_data['action']
        }
        raw_data['subtask'] = create_initialprompt_start(sample)
        
        with open(save_path+f'variation{vari}.json', 'w') as json_file:
            json.dump(raw_data, json_file, indent=4)
            print(task_id, vari,"   done")


        # exit()