import numpy as np

np.set_printoptions(precision=3)

def evaluate(pre, target, device):
    # pre [batch size, embeding dim]
    # target [batch size]
    # emb [num embedings, embeding size]
    tot_hit1 = 0
    tot_hit3 = 0
    tot_hit10 = 0
    tot_MRR = 0.0
    tot_MR = 0
#     pre = emb.unsqueeze(1) - pre.unsqueeze(0)  # pre = [num embedings,batch size, embeding dim]
#     pre = pre.transpose(0, 1)  # pre = [batch size, num embedings, embeding dim]
#     temp = torch.ones(pre.shape[0], pre.shape[2], 1) # tmep = [batch size, embeding dim, 1]
#     res = torch.bmm(pre.to(device), temp.to(device)) # res = [batch size, num embedings, 1]
#     res = res.reshape(res.shape[0], res.shape[1]) # res = [batch size, num embedings]

#     pre = (-res).cpu().detach().numpy()
    pre = pre.cpu().detach().numpy()
    for i,each in enumerate(pre):
        t = each[target[i]]
        each.sort()
        t_idx = np.where(each==t)[0][0]
        if t_idx < 1:
            tot_hit1 += 1
        if t_idx < 3:
            tot_hit3 += 1
        if t_idx < 10:
            tot_hit10 += 1
        tot_MRR += 1.0/(t_idx+1)
        tot_MR += t_idx+1
    return tot_MRR, tot_MR, tot_hit1, tot_hit3, tot_hit10