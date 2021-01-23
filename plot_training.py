import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%matplotlib inline
plt.rcParams["figure.figsize"] = (8,5)
filename = "training_info.csv"
eval_name = "Jamesbond-v0-Eval.csv"
n_steps = 10
f = pd.read_csv(filename)
g = pd.read_csv(eval_name)
mean = f[" return"].rolling(n_steps).mean()
deviation = f[" return"].rolling(n_steps).std()
under_line = (mean-deviation)
over_line = (mean+deviation)
plt.plot(f["training_step"], mean, linewidth=2, label="training return")
plt.fill_between(f["training_step"], under_line, over_line, color='b', alpha=0.1)
plt.scatter(g["steps"], g["return"], linewidth=2, color="r", label="evaluation return")
plt.xlabel("Number of steps")
plt.ylabel("Return")
plt.legend()
plt.show()