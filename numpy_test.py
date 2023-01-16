import numpy as np
A = np.array([[1,2],[3,4],[5,6]])
A = (A - A.mean(axis = 0))/A.std(axis = 0)
print(A)
