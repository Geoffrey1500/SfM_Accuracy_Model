import numpy as np


x = []
for i in range(5000):
    L_ = np.random.normal(0, 1)
    x.append(L_)
x = np.array(x)
print(x)

print(np.average(x), np.std(x))
