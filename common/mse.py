y = 500

x = y / 2

lr = 0.001
epochs = 10000

for epoch in range(epochs):
    if(abs(x * x - y) <= 1e-5):
        break
    d = 4 * x * (x * x - y)
    if(abs(d) > 1e3):
        if(d > 0): d = 1e3
        else: d = -1e3
    x = x - lr * d
print(x)

