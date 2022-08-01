import numpy as np


x = np.array([[1,2,3,9],[-2,-4,-6,18],[4,8,0,0]])
x_mn = np.mean(x, axis=0)

def Sigma (sample:np.array, sample_mean):
    total =  np.matmul((sample-mean), (sample-mean).T)
    print(total)
    return total

sum = 0
for row in x_1:
    print(row)
    sum += Sigma(row)

sum = (1/3) * sum

print(sum)


x = ((1-1)^2 + (-2-1)^2 + (4-1)^2)/2 & ((2--2)^2 + (-4--2)^2 + (8--2)^2)/2 & ((3--1)^2 + (-6--1)^2 + (0--1)^2)/2 & ((9--3)^2 + (-18--3)^2 + (0--3)^2)/2