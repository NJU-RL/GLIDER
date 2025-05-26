import re
def extract_action(llm_output_text, action_space):
    """
    from text extract action
    parameters:
    - llm_output_text (str):text
    - action_space (list): action list
    """
    llm_output_text = llm_output_text.strip().replace("go to", "go").replace("the ", " ").replace("into", "in").replace("  "," ").replace("\n",".")
    
    action_patterns = [action.replace("OBJ", r"[\w\s]+").strip() for action in action_space]
    
    action_regexes = [re.compile(rf"\b{pattern}\b", re.IGNORECASE) for pattern in action_patterns]
    
    extracted_actions = []
    for regex in action_regexes:
        matches = regex.findall(llm_output_text)
        extracted_actions.extend([match.strip() for match in matches])

    if extracted_actions:
        longest_action = max(extracted_actions, key=len)
        return longest_action
    else:
        return ""
    

def extract_action_done(text):
    match = re.match(r'(.*?)\s*;\s*(true|false)', text, re.I)
    if match:
        action, done = match.groups()
        return action.strip(), True if done.strip().lower() == 'true' else False
    return "None", False