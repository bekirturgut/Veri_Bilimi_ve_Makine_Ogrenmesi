import numpy as np
import matplotlib.pyplot as plt

age_list = [15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]
weight_list = [51,52,53,54,55,68,57,75,59,60,61,99,63,64,65,66,67,68,69,70,71,72]

#plt.plot(age_list,weight_list)
#plt.xlabel("age")
#plt.ylabel("weight")
#plt.title("Age and Weight")
#plt.show()

np_age_list = np.array(age_list)
np_weight_list = np.array(weight_list)

#plt.plot(np_age_list,np_age_list**8, "b--") # - düz , * nokta , *- noktalı çizgi , -- kesikli çizgi
#plt.show()

plt.subplot(1,2,1) #iki grafiği bu sayede aynı tabloya sığdırdık
plt.plot(np_weight_list,np_weight_list**8, "b--")
plt.subplot(1,2,2)
plt.plot(np_age_list,np_age_list**8,"r--")
plt.show()