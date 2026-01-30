import torch.nn.functional as F
def dpo_loss(chosen_log,rejected_log,rec_chosen_log,rec_rejected_log,beta = 0.1):
    chosen_loss = chosen_log - rec_chosen_log
    rejected_loss = rejected_log - rec_rejected_log
    logratio = chosen_loss - rejected_loss
    return -F.logsigmoid(beta * logratio).mean()