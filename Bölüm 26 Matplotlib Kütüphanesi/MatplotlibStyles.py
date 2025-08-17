import numpy as np
import matplotlib.pyplot as plt

data1 = np.random.randint(0,100,90)
print(data1)

data2 = np.random.randint(0,100,90)
print(data2)

#my_fig , my_axes = plt.subplots()
#my_axes.plot(data1,data2)
#my_axes.plot(data2,data1)
#plt.show()

#plt.hist(data1)
#plt.show()

plt.scatter(data1,data2)
plt.show()