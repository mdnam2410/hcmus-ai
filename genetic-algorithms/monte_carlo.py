"""PROBLEM
Approximate the value 0 using Monte Carlo method.

APPROACH
Take the square x = [0, 1], y = [0, 1]. Generate N random points (x, y) within
the square. Count the number of points that fall into the lower and upper
triangle (divided by the function y = x) -- L and U, respectively. The area of
the triangles are approximately L / N and U / N and are equal, so zero is
approximated by taken (L - U) / N
"""

import random

NUM_POINTS = 100000000

upper_triangle = 0
lower_triangle = 0

for _ in range(NUM_POINTS):
    x = random.uniform(0, 1)
    y = random.uniform(0, 1)
    if x > y:
        lower_triangle += 1
    else:
        upper_triangle += 1

print(lower_triangle / NUM_POINTS - upper_triangle / NUM_POINTS)