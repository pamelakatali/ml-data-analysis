import os
import torch
from torch import nn
from torch.utils.data import Dataset

def ebay_loss(inp, target):
    pE = 0.4
    pL = 0.6

    diff = torch.sub(target, inp)

    diff_e = diff.clone()
    diff_e[diff_e > 0] = 0
    diff_e = pE * torch.abs(torch.sum(diff_e))

    diff_l = diff.clone()
    diff_l[diff_l < 0] = 0
    diff_l = pL * torch.sum(diff_l)

    N = inp.size(dim=0)

    output = (1.0/(N+0.00000001))*torch.add(diff_e, diff_l)

    alph = 3
    scaled_err = alph*(1.0/(N+0.00000001))*torch.sum(torch.abs(torch.sub(torch.log(torch.clamp(target.clone(), min=0.001)), torch.log(torch.clamp(inp.clone(), min=0.001)))))
    #print(output2)
    return output+scaled_err

'''

# Inherit from Function
class EbayLoss(torch.autograd.Function):
    # Note that both forward and backward are @staticmethods
    # bias is an optional argument
    @staticmethod
    def forward(ctx, inp, target):
      pE = 0.4
      pL = 0.6


      diff = torch.sub(target, inp)

      diff_e = diff.clone()
      diff_e[diff_e > 0] = 0
      diff_e = pE * torch.sum(diff_e)
      

      diff_l = diff.clone()
      diff_l[diff_l < 0] = 0
      diff_l = pL * torch.abs(torch.sum(diff_l))
      

      N = inp.size(dim=0)

      output = (1.0/(N+0.00000001))*torch.add(diff_e, diff_l)
      
      #print(output)
      ctx.save_for_backward(inp, target, diff)
      return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        pE = 0.4
        pL = 0.6
        
        inp, target, diff = ctx.saved_tensors
        N = inp.size(dim=0)

        diff_e = diff.clone()
        diff_e[diff_e > 0] = 0
        diff_e[diff_e < 0] = 1
        diff_e = pE *  torch.abs(diff_e)

        diff_l = diff.clone()
        diff_l[diff_l < 0] = 0
        diff_l[diff_l > 0] = 1
        diff_l = pL * (-1.0) * torch.abs(diff_l)

        grad_input = (1.0/(N+0.00000001))*torch.add(diff_e, diff_l)


        return grad_input, None
'''