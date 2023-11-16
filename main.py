from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
import numpy as np
import math
from Population import Population
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def func(vector, n):
    return -20.0 * exp(-0.2 * sqrt(1 / n * sum(i * i for i in vector))) - exp(
            1 / n * sum(math.cos(i * 2 * pi) for i in vector)) + e + 20


print(Population(50, 1000, 0.5, 0.01, func, 2).start())

