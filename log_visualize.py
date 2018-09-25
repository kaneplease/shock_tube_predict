import pandas as pd
import matplotlib.pyplot as plt

# Read file
df = pd.read_csv('log.csv')

# Set values
x = df['epoch'].values
y0 = df['train_loss'].values

# Set background color to white
fig = plt.figure()
fig.patch.set_facecolor('white')

# Plot lines
plt.xlabel('epoch')
plt.plot(x, y0, label='train_loss')
plt.legend()

plt.savefig('log_viz.png')

# Visualize
plt.show()