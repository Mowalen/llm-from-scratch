"""
y_pred: shape (batch_size, num_classes) — softmax 概率
y_true: shape (batch_size, num_classes) — one-hot 编码标签
"""
import numpy as np
def cross_entropy(y_pred, y_true):
    epsilon = 1e-12 # 要对 log 做一个 clip 来防止过高或过低
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    losses = -np.sum(y_true * np.log(y_pred), axis=1)
    return np.mean(losses)  # 解释代码