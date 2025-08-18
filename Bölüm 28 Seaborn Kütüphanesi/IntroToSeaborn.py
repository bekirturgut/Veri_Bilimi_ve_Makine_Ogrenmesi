import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = np.random.randint(0,10000,size=(400,2))
df = pd.DataFrame(data, columns=["weight", "height"])

sns.set_style('whitegrid')
sns.scatterplot(x="weight", y="height", data=df)

plt.xlabel('weight')
plt.ylabel('height')
plt.title("selammlarrrr")
plt.show()