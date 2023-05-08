import torch

def score_loss(y_pred, y_true, alpha1=13, alpha2=10):   
    batch_size = y_pred.size(0)   
    e = y_pred - y_true
    e = e.view(-1, 1).clamp(-100, 100)
    e1 = e[e<=0]
    e2 = e[e>0]
    s1 = (torch.exp(-(e1 / alpha1)) - 1)
    s2 = (torch.exp((e2 / alpha2)) - 1)
    score = torch.cat((s1, s2))
    #score = score[torch.isfinite(score)]
    #score = score[torch.isfinite(score)].mean()
    score = score[torch.isfinite(score)].mean()*batch_size
    return score

def score_focal_loss(y_pred, y_true, alpha1=13, alpha2=10):   
    #batch_size = y_pred.size(0) 
    e = y_pred - y_true
    e = e.view(-1, 1).clamp(-100, 100)
    e1 = e[e<=0]
    e2 = e[e>0]
    s1 = (torch.exp(-(e1 / alpha1)) - 1)
    s2 = (torch.exp((e2 / alpha2)) - 1)
    score = torch.cat((s1, s2))
    #score = score[torch.isfinite(score)]
    #score = score[torch.isfinite(score)].mean()
    #score = score[torch.isfinite(score)].mean()*batch_size
    score = score[torch.isfinite(score)]
    return score
