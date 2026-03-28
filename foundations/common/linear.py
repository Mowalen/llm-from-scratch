x_data = [1.0, 2.0, 3.0, 4.0]
y_data = [3.0, 5.0, 7.0, 9.0]
n = len(x_data)

w = 0.0
b = 0.0
lr = 0.01

ephocs = 1000

for epoch in range(ephocs):
    w_grad = 0.0
    b_grad = 0.0
    for i in range(n):
        y_pred = w * x_data[i] + b
        error = y_pred - y_data[i]
        w_grad += -2 * error * x_data[i]
        b_grad += -2 * error * 1.0

    w -= (w_grad / n) * lr
    b -= (b_grad / n) * lr
    if(epoch % 100 == 0):
        loss = sum([(y - (w*x + b))**2 for x, y in zip(x_data, y_data)]) / n
        print(f"Epoch {epoch}: Loss={loss:.4f}")
    