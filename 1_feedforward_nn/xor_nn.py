import numpy as np 
#importing numpy 

X= np.array ([0,0 ], [0,1],[1,0],[1,1])# x is my input matrix with 4 example with 2 inputs each XOR truth table like 
Y = np.array([0],[1],[1],[0]) # y is the target for each input XOR(0,0)=0, XOR(0,1)=1, etc.
# intiallinzing 
np.random.seed(42)
w1 = np.random.random((2,1))