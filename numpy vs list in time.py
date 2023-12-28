import numpy as np

arr = np.arange(1000000)
pylist = list(range(1000000))

%time for item in range(10): [item * 3 for item in pylist] # for list
%time for item in range(10): arr = arr * 3 # for numpy

# numpy wall time is less than list wall time

print(type(arr))
