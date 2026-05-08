########################################################################
# Script to compare the ReLU and Softplus activation functions.
########################################################################
import numpy as np
import matplotlib.pyplot as plt

def softplus(x, beta=1.0):
    return (1.0 / beta) * np.log(1 + np.exp(beta * x))

x = np.linspace(-3, 3, 500)

plt.figure()
plt.plot(x, np.maximum(0, x), label='ReLU', color='black', alpha=0.3, linewidth=4)
# plt.plot(x, softplus(x, beta=0.5), label='Beta = 0.5 (Very Smooth)')
# plt.plot(x, softplus(x, beta=1.0), label='Beta = 1.0 (Standard)')
plt.plot(x, softplus(x, beta=5.0), label='Beta = 5.0')

# plt.title(r'Softplus Smoothness controlled by $\beta$')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()