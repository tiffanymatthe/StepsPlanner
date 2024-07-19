import numpy as np
import matplotlib.pyplot as plt

### 14 DEG
next_steps = [
    [-1.60000000e-01,  1.35000000e+00,  8.26636589e-17,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  1.57079633e+00,  1.00000000e+00],
    [1.60000000e-01, 2.00000000e+00, 1.22464680e-16, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.57079633e+00, 0.00000000e+00],
    [-1.60000000e-01,  2.65000000e+00,  1.62265701e-16,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  1.57079633e+00,  1.00000000e+00],
    [1.60000000e-01, 3.30000000e+00, 2.02066722e-16, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.57079633e+00, 0.00000000e+00],
]

next_next_steps = [
    [-2.00191594e-03,  2.01939973e+00,  1.22464680e-16 -2.44346095e-01, 0.00000000e+00,  0.00000000e+00,  1.32645023e+00,  0.00000000e+00],
    [-3.12496548e-01,  2.59198472e+00,  1.62265701e-16 -2.44346095e-01, 0.00000000e+00,  0.00000000e+00,  1.32645023e+00,  1.00000000e+00],
    [-2.00191594e-03,  3.31939973e+00,  2.02066722e-16 -2.44346095e-01, 0.00000000e+00,  0.00000000e+00,  1.32645023e+00,  0.00000000e+00],
    [-3.12496548e-01,  3.24198472e+00,  2.41867743e-16 -2.44346095e-01, 0.00000000e+00,  0.00000000e+00,  1.32645023e+00,  1.00000000e+00],
]

# ### 0 DEG
# next_steps = [
#     [-1.60000000e-01,  1.35000000e+00,  8.26636589e-17,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  1.57079633e+00,  1.00000000e+00],
#     [1.60000000e-01, 2.00000000e+00, 1.22464680e-16, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.57079633e+00, 0.00000000e+00],
#     [-1.60000000e-01,  2.65000000e+00,  1.62265701e-16,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  1.57079633e+00,  1.00000000e+00],
#     [1.60000000e-01, 3.30000000e+00, 2.02066722e-16, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.57079633e+00, 0.00000000e+00],
# ]

# next_next_steps = [
#     [1.60000000e-01, 2.00000000e+00, 1.22464680e-16, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.57079633e+00, 0.00000000e+00],
#     [-1.60000000e-01,  2.65000000e+00,  1.62265701e-16,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  1.57079633e+00,  1.00000000e+00],
#     [1.60000000e-01, 3.30000000e+00, 2.02066722e-16, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.57079633e+00, 0.00000000e+00],
#     [-1.60000000e-01,  3.30000000e+00,  2.41867743e-16,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  1.57079633e+00,  1.00000000e+00],
# ]

def plllt(steps, start_index = 2):
    plt.quiver([x[0] for x in steps], [x[1] for x in steps], np.cos([x[-2] for x in steps]), np.sin([x[-2] for x in steps]))
    plt.scatter([x[0] for x in steps], [x[1] for x in steps])
    for i in range(len(steps)):
        plt.annotate(i+start_index, (steps[i][0], steps[i][1]))

plllt(next_steps)
plllt(next_next_steps, start_index = 3)

plt.axis('square')
plt.show()