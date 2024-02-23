import numpy as np
#original python list
list1= [1,2,3]

#Numpy - Numeric Python
# ndarray = n -dimensional array
np1 = np.array([0,1,2,3,4,5,6,7,8,9])
print(np1)
print(np1.shape)

#Range
np2 = np.arange(10)
print(np2)
#0~10 2씩 증가 
np3 = np.arange(0,10,2)
print(np3)

np4 = np.zeros(10)
print(np4)

np5 = np.zeros((2,10))
print(np5)
#2 by 10 array with 6 
np6 = np.full((2,10),6)
print(np6)
my_list = [1,2,3,4,5]
print(np.array(my_list))
##############################
#Slicing array
np7 = np.array([1,2,3,4,5,6,7,8,9])
print(np7[1:5])  #[2 3 4 5]
print(np7[-3:-1]) #[7 8]
#Slicing 2-D Array
np8 = np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]])
print(np8[0:2,2:4])
#np ufunc함수 사용  -->  np.함수(array 이름)
print(np.absolute(np3))
print(np.sqrt(np4))
# view
np1 = np.array([0,1,2,3,4,5,6,7,8,9])
np_view = np1.view()
print(np_view) #[0 1 2 3 4 5 6 7 8 9]
np1[0] = 1
#np1 을 바꾸면 view인 np2 도 바뀜  np2 는 view of np1 
print(np_view) #[1 1 2 3 4 5 6 7 8 9]
#view를 바꿔도 original이 바뀜

np_copy = np1.copy()
print(np_copy) #[1 1 2 3 4 5 6 7 8 9]
np1[0] = 3
#np1 을 바꾸면 copy 였던 np_copy는 변하지 않음 
print(np_copy) #[1 1 2 3 4 5 6 7 8 9]

#reshape
np1 = np.array([0,1,2,3,4,5,6,7,8,9,10,11])
np3 = np1.reshape(3,4)
print(np3)
#[[ 0  1  2  3]
# [ 4  5  6  7]
# [ 8  9 10 11]]

#filter using numpy
np1 = np.array([0,1,2])
x = [True, False, False]
print(np1[x])  #[0]

filtered = [] 
for element in np1:
    if element %2 ==0:
        filtered.append(True)
    else:
        filtered.append(False)
print(np1[filtered])
#이렇게 하면 귀찮고 오래 걸리니까
filtered = np1 % 2 ==0 #해도 똑같은 결과가 나옴


