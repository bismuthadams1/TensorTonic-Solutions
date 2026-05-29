import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    # Your code here

    # W shape (d_model, d_model)
    # Q shape (batch, seq_len, d_model)
    # Q * W  = (batch, seq_len, d_model)x(d_model,d_model) = ( batch, seq_len, d_model)

    # then reshape Q_W  to (batch, seq_len, h_n, d_k) where h_n * d_k = d_model
    # we then want to multiply  Q (batch,n_heads,seq_len,d_k) * K^T (batch, n_heads, d_k, seq_len)
    # yielding (batch, n_heads, seq_len, seq_len)
    # then multiply by reshaped V (batch, n_heads, seq_len, d_k)
    d_k =  int(W_q.shape[0] / num_heads)
    
    batch_size = Q.shape[0]
    seq_len = Q.shape[1]


    Q_heads = (Q @ W_q).reshape(batch_size, seq_len, num_heads, d_k)
    K_heads = (K @ W_k).reshape(batch_size, seq_len, num_heads, d_k)
    V_heads = (V @ W_v).reshape(batch_size, seq_len, num_heads, d_k)

    qk_mult = np.einsum('bind,bjnd->bnij', Q_heads, K_heads)/np.sqrt(d_k) # multiple seq_len by seq len. Give us batch_size,num_heads,seq_len,seq_len

    qk_mult = softmax(qk_mult) # softmax this

    qkd_v_mult = np.einsum('bnij,bjnd->bind', qk_mult, V_heads) # multiply seq-len axis with each other to yield out original shape    
    qkd_v_mult_concat = qkd_v_mult.reshape(batch_size, seq_len, num_heads * d_k)  #concat and then multiple by W_0(d_model, d_model).
    # Equivalent einsum
    # qkd_v_mult_concat = np.einsum('bind->bid',qkd_v_mult)
    # qkd_v_mult_concat_W_0 = np.einsum('bid,Dd->bid', qkd_v_mult_concat, W_0)
    out = np.einsum('bid,df->bif', qkd_v_mult_concat, W_o)
    
    return out