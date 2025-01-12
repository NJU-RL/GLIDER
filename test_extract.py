import torch
from torch.nn.utils.rnn import pad_sequence

def extract_valid_action_probs(action_log_probs, action_masks):
    """
    Args:
        action_log_probs: (batch, seq_len)
        action_masks: (batch, seq_len), 1表示action token位置
    Returns:
        valid_action_probs: (batch, max_actions) 每个action的平均log概率
        mask: (batch, max_actions) padding位置标记
    """
    batch_size = action_log_probs.size(0)
    
    # 首先找出每个batch中有多少个完整的action
    action_probs_list = []
    for i in range(batch_size):
        # 找出所有1的位置
        action_positions = torch.where(action_masks[i] == 1)[0]
        
        # 将连续的positions分组
        action_groups = []
        current_group = []
        for pos in action_positions:
            if not current_group or pos == current_group[-1] + 1:
                current_group.append(pos)
            else:
                action_groups.append(current_group)
                current_group = [pos]
        if current_group:
            action_groups.append(current_group)
        
        # 对每组计算平均log概率
        action_probs = []
        for group in action_groups:
            group_probs = action_log_probs[i, group]
            avg_prob = group_probs.sum() / len(group)
            action_probs.append(avg_prob)
        
        action_probs_list.append(torch.tensor(action_probs, device=action_log_probs.device))
    
    # 使用pad_sequence进行填充
    padded_probs = pad_sequence(action_probs_list, batch_first=True, padding_value=0)
    return padded_probs
# 使用示例

# 创建测试数据
action_log_probs = torch.tensor([
    [-1.0, -2.0, -1.5, -0.5, -0.8, 0.0],
    [-0.5, -1.0, -1.5, -2.0, -0.3, -0.4]
])
action_masks = torch.tensor([
    [0, 1, 1, 0, 1, 0],  # 第一个样本有两个action，第一个长度为2，第二个长度为1
    [0, 0, 0, 1, 0, 0]   # 第二个样本有两个action，第一个长度为2，第二个长度为1
])

valid_probs = extract_valid_action_probs(action_log_probs, action_masks)
print("Valid action probs:", valid_probs)
# print("Mask:", mask)
