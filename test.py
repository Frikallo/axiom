import numpy as np
import time

a = np.random.randn(1000, 1000)
b = np.random.randn(1000, 1000)
a = a.astype(np.float16)
b = b.astype(np.float16)

start = time.time()
c = np.matmul(a, b)
end = time.time()
print(f"Matmul of 1000x1000 tensors on CPU took {(end - start) * 1000}ms")