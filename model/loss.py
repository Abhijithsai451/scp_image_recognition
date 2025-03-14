import torch.nn.functional as F


def nll_loss(output, target):
    print(' [DEBUG] loss.py in nll_loss()\n')
    return F.nll_loss(output, target)
