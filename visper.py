from perceptron import Perceptron
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

ref = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
table = pd.read_csv(ref, header=None, encoding="utf-8")

print("-" * 50)
print(table.tail(10))
print("-" * 50)

y = table.iloc[0:100, 4].values
x = table.iloc[0:100, [0,2]].values
y = np.where(y == 'Iris-setosa', 0, 1)

plt.scatter(x[:50,0], x[:50,1], color='red', marker='o', label='Setosa')
plt.scatter(x[50:100,0], x[50:100,1], color='blue', marker='s', label='Versicolor')
plt.xlabel('Len Sepals')
plt.ylabel('len Petal')
plt.legend(loc='upper left')

plt.show()

ppn = Perceptron(eta=0.1, n_iter=15)
ppn.fit(x, y)
plt.plot(range(1, len(ppn.errors) + 1), ppn.errors, marker='o')
plt.xlabel('Era')
plt.ylabel('Count update')

plt.show()


