import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # Q shape (b,s,d) (1,1,1)
    # K shape (b,s,d) (1,1,3) 
    
    # V shape (b,s,d)
    dot_prod = torch.einsum('bik,bjk->bij', Q,K)
    # dot_prod output (b,s,s) (1,1,3)
    d_k = K.shape[-1]
    d_v = V.shape[-1] 
    
    scaled_dot_prod = F.softmax(dot_prod / math.sqrt(d_k), dim=-1)

    result = torch.einsum('bji,bik->bjk', scaled_dot_prod, V)#.squeeze(-1)
    # (1,1,3) * (1,1,3) = (1,3,3)
    
    return result
    