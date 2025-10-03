import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from adaline import AdalineCD

ref = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
table = pd.read_csv(ref, header=None, encoding="utf-8")

y = table.iloc[0:100, 4].values
x = table.iloc[0:100, [0,2]].values
y = np.where(y == 'Iris-setosa', 0, 1)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

ada1 = AdalineCD(n_iter=15, eta=0.05).fit(x, y)
ax[0].plot(range(1, len(ada1.losses) + 1),
            np.log10(ada1.losses), marker='o')
ax[0].set_xlabel('Era')
ax[0].set_ylabel('Log')
ax[0].set_title('Adaline - speed learning 0.05')

ada2 = AdalineCD(n_iter=15, eta=0.001).fit(x, y)
ax[1].plot(range(1, len(ada2.losses) + 1),
            np.log10(ada2.losses), marker='o')
ax[1].set_xlabel('Era')
ax[1].set_ylabel('Log')
ax[1].set_title('Adaline - speed learning 0.001')

plt.show()

x_std = np.copy(x)
x_std[:,0] = (x[:,0] - x[:,0].mean()) / x[:,0].std()
x_std[:,1] = (x[:,1] - x[:,1].mean()) / x[:,1].std()
ada_gd = AdalineCD(n_iter=20, eta=0.5)
ada_gd.fit(x_std, y)

plt.scatter(x[:50,0], x[:50,1], color='red', marker='o', label='Setosa')
plt.scatter(x[50:100,0], x[50:100,1], color='blue', marker='s', label='Versicolor')
plt.title('Adaline - GD')
plt.xlabel('Len Sepals')
plt.ylabel('len Petal')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

plt.plot(range(1, len(ada_gd.losses) + 1),
            np.log10(ada_gd.losses), marker='o')
plt.xlabel('Era')
plt.ylabel('Log')
plt.title('Adaline - speed learning 0.5')
plt.show()