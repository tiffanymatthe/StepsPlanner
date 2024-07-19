import numpy as np
import matplotlib.pyplot as plt
import re

text = """
Creating temp states for 1
Index: 1 Centered x y is 0.4349348941332579, 1.1830441365603062
Index: 1 for -0.7330382858376184 and 0.0 and 0.65
array([1.60000000e-01, 7.00000000e-01, 4.28626380e-17, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 1.57079633e+00, 0.00000000e+00])
array([ 3.16031722e-01,  1.29010503e+00,  8.26636589e-17, -7.33038286e-01,
        0.00000000e+00,  0.00000000e+00,  8.37758041e-01,  1.00000000e+00])
Creating temp states for 2
Index: 2 Centered x y is 0.4349348941332579, 1.8330441365603063
Index: 2 for -0.7330382858376184 and 0.0 and 0.65
array([-1.60000000e-01,  1.35000000e+00,  8.26636589e-17,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  1.57079633e+00,  1.00000000e+00])
array([ 5.53838066e-01,  1.72598324e+00,  1.22464680e-16, -7.33038286e-01,
        0.00000000e+00,  0.00000000e+00,  8.37758041e-01,  0.00000000e+00])
Creating temp states for 3
Index: 3 Centered x y is 0.4349348941332579, 2.483044136560306
Index: 3 for -0.7330382858376184 and 0.0 and 0.65
array([1.60000000e-01, 2.00000000e+00, 1.22464680e-16, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 1.57079633e+00, 0.00000000e+00])
array([ 3.16031722e-01,  2.59010503e+00,  1.62265701e-16, -7.33038286e-01,
        0.00000000e+00,  0.00000000e+00,  8.37758041e-01,  1.00000000e+00])
Creating temp states for 4
Index: 4 Centered x y is 0.4349348941332579, 3.133044136560306
Index: 4 for -0.7330382858376184 and 0.0 and 0.65
array([-1.60000000e-01,  2.65000000e+00,  1.62265701e-16,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  1.57079633e+00,  1.00000000e+00])
array([ 5.53838066e-01,  3.02598324e+00,  2.02066722e-16, -7.33038286e-01,
        0.00000000e+00,  0.00000000e+00,  8.37758041e-01,  0.00000000e+00])
Creating temp states for 5
Index: 5 Centered x y is 0.4349348941332579, 3.783044136560306
Index: 5 for -0.7330382858376184 and 0.0 and 0.65
array([1.60000000e-01, 3.30000000e+00, 2.02066722e-16, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 1.57079633e+00, 0.00000000e+00])
array([ 3.16031722e-01,  3.89010503e+00,  2.41867743e-16, -7.33038286e-01,
        0.00000000e+00,  0.00000000e+00,  8.37758041e-01,  1.00000000e+00])
"""

next_steps = []
next_next_steps = []
next_next_steps_centered = []

# Regular expression to match arrays
array_pattern = re.compile(r'array\(\[([^\]]+)\]\)', re.MULTILINE)
centered_pattern = re.compile(r'Centered x y is ([\d\.\-e]+), ([\d\.\-e]+)', re.MULTILINE)

# Find all arrays in the text
arrays = array_pattern.findall(text)
centered_xy = centered_pattern.findall(text)

# Split and convert string arrays to numpy arrays
next_steps = [np.fromstring(arrays[i], sep=',') for i in range(0, len(arrays), 2)]
next_next_steps = [np.fromstring(arrays[i+1], sep=',') for i in range(0, len(arrays), 2)]

next_next_steps_centered = []

for i, (cx, cy) in enumerate(centered_xy):
    modified_array = next_next_steps[i].copy()
    modified_array[0] = float(cx)
    modified_array[1] = float(cy)
    next_next_steps_centered.append(modified_array)

# Print the results
print("Next Steps:")
for arr in next_steps:
    print(arr)

print("\nNext Next Steps:")
for arr in next_next_steps:
    print(arr)

print("\nModified Next Next Steps:")
for arr in next_next_steps_centered:
    print(arr)

def plllt(steps, start_index = 1):
    plt.quiver([x[0] for x in steps], [x[1] for x in steps], np.cos([x[-2] for x in steps]), np.sin([x[-2] for x in steps]))
    plt.scatter([x[0] for x in steps], [x[1] for x in steps])
    for i in range(len(steps)):
        plt.annotate(i+start_index, (steps[i][0], steps[i][1]))

plllt(next_steps)
plllt(next_next_steps, start_index = 2)
plllt(next_next_steps_centered, start_index = 2)

plt.axis('square')
plt.show()