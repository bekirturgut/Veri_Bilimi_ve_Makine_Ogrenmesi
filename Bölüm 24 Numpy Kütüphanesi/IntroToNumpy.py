from xml.etree.ElementTree import tostring

import numpy as np

my_list = [10,20,30,40]
print(type(my_list))
print(my_list,end="\n\n---------------------------------------------------------\n\n")

my_array = np.array(my_list)
print(type(my_array))
print(my_array,end="\n\n")
print(np.random.random((2,2)),end="\n\n---------------------------------------------------------\n\n")

#Array aritmetiği

mylist1 = [1,2]
mylist2 = [2,3]
print(mylist1+mylist2,end="\n\n---------------------------------------------------------\n\n")

numpy_array1 = np.array(mylist1)
numpy_array2 = np.array(mylist2)
print(numpy_array1+numpy_array2)
print(numpy_array1-numpy_array2)
print(numpy_array1/numpy_array2)
print(numpy_array1*5)
print(numpy_array1*numpy_array2,end="\n\n---------------------------------------------------------\n\n")

#arange & indexing

print(np.arange(0,10))
print(numpy_array1[-2])
print(np.random.randint(0,10,2),end="\n\n---------------------------------------------------------\n\n")

#Matrix

my_matrix = np.array([[1,2],[3,4],[5,6],[7,8]])
print(my_matrix)
print(my_matrix.sum())
print(np.ones((3,3)),end="\n\n---------------------------------------------------------\n\n")

# matrix aritmetiği

my_matrix1 = np.array([[10,20],[30,40]])
my_matrix2 = np.array([[5,15],[25,35]])
print(my_matrix1+my_matrix2,end="\n###########\n")
print(my_matrix1-my_matrix2,end="\n###########\n")
print(my_matrix1/my_matrix2,end="\n###########\n")
print(my_matrix1/my_matrix2,end="\n###########\n")
print(my_matrix1.shape,end="\n\n---------------------------------------------------------\n\n") # kaç sıra kaç kolon olduğunu gösterir

# Dot product örneği

my_matrix3 = np.array([[1,2,5,6],[3,4,7,8]])
my_matrix4 = np.array([[10,20]])
my_matrix5 = my_matrix4.dot(my_matrix3)
print(my_matrix3,end="\n###########\n")
print(my_matrix4,end="\n###########\n")
print(my_matrix5,end="\n\n---------------------------------------------------------\n\n")

# örnek

new_array = np.random.randint(0,100,20)
print(new_array,end="\n###########\n")
print(new_array > 25,end="\n###########\n")
print(new_array[new_array > 25],end="\n###########\n")
new_matrix = np.array([[1,2,5,6],[3,4,7,8],[9,10,11,12]])
print(new_matrix,end="\n###########\n")
print(new_matrix.transpose(),end="\n###########\n")