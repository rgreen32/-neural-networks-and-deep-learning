import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

fig, ax = plt.subplots()
y = np.array([random.randrange(0, y+1) for y in range(20)])
x = np.arange(20)
ax.scatter(x, y)


