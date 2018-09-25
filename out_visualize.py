import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('true.csv', names=['num1', 'num2', 'num3'])
df2 = pd.read_csv('out2.csv', names=['num1', 'num2', 'num3'])
plt.plot(range(0,50),df['num3'],marker="o", label = 'Analytical Result')
plt.plot(range(0,50),df2['num3'],marker="o", label = 'NN Result')
plt.title("p_graph")
plt.legend()
plt.savefig('train_out.png')
plt.show()
