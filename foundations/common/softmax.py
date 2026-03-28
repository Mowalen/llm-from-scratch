import numpy as np

def softmax(logits, T=1.0): # 不考虑维度的版本
    logits = np.array(logits) / T
    exps = np.exp(logits)
    return exps / np.sum(exps)
  
def softmax(logits, T=1.0):  # 考虑维度和数值溢出的版本
    # logits: shape (batch_size, num_classes)
    logits_stable = logits - np.max(logits, axis=1, keepdims=True)  # 防止数值溢出
    exps = np.exp(logits_stable / T)  # 在指数运算前除以温度 T
    return exps / np.sum(exps, axis=1, keepdims=True)
