import numpy as np
np.set_printoptions(precision=3)

def evaluate(pre, target, emb, device):
    # pre [batch size, embeding dim]
    # target [batch size]
    # emb [num embedings, embeding size]
    tot_hit1 = 0
    tot_hit3 = 0
    tot_hit10 = 0
    tot_MRR = 0.0
    tot_MR = 0

#     pre = (-res).cpu().detach().numpy()
    pre = pre.cpu().detach().numpy()
    for i,each in enumerate(pre):
        max_idx = np.argmax(each)
        m = np.argpartition(each, max_idx)
#         hit3 = m[:3]
#         hit5 = m[:5]
#         hit10 = m[:10]
        if target[i] == max_idx:
            tot_hit1 += 1
        if target[i] in m[:3]:
            tot_hit3 += 1
        if target[i] in m[:10]:
            tot_hit10 += 1
            
        for j, idx in enumerate(m):
            if(idx==target[i]):
                tot_MRR += 1.0/(j+1.0)
                tot_MR += j+1
                break
    
    return tot_MRR, tot_MR, tot_hit1, tot_hit3, tot_hit10