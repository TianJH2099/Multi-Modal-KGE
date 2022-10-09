from pickletools import TAKEN_FROM_ARGUMENT1
import numpy as np
import torch

np.set_printoptions(precision=3)

def evaluate(pre, target, device):
    # pre [batch size, embeding dim]
    # target [batch size]
    tot_hit1 = 0
    tot_hit3 = 0
    tot_hit10 = 0
    tot_MRR = 0.0
    tot_MR = 0

    # pre = -pre.cpu().detach().numpy()
    for i,each in enumerate(pre):
        t = target[i]
        sort_value, sort_key = torch.sort(each, dim=0, descending=True)
        sort_key = sort_key.cpu()
        # each.sort(descending=True)
        t_idx = np.where(sort_key==t)[0][0]
        if t_idx < 1:
            tot_hit1 += 1
        if t_idx < 3:
            tot_hit3 += 1
        if t_idx < 10:
            tot_hit10 += 1
        tot_MRR += 1.0/(t_idx+1)
        tot_MR += t_idx+1
    return tot_MRR, tot_MR, tot_hit1, tot_hit3, tot_hit10