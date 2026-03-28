# """
# y_pred: shape (batch_size, num_classes) — softmax 概率
# y_true: shape (batch_size, num_classes) — one-hot 编码标签
# ""
import numpy as np

def cross_entropy(y_pred, y_true):
    e = 1e-12
    y_pred = np.clip(y_pred,e,1.0 - e)
    return np.mean(-np.sum(y_true * np.log(y_pred), axis=1))