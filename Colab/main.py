import matplotlib.pyplot as plt
import numpy as np
from sympy.physics.control.control_plots import matplotlib

test = [0.2149, 4.6212, 5.4624, 5.4781, 5.5165, 5.7785, 5.7192, 5.7152, 5.7646, 5.7329, 5.7629, 5.8702, 5.8506, 5.9400, 5.9724, 5.7612,
        5.9765, 5.9215, 5.9913, 5.8764, 5.7662, 5.9397, 6.0107, 6.0350, 6.0026, 5.9036, 5.8948, 5.8789, 6.0374, 6.0688, 5.9658, 6.0879,
        6.1142, 6.0726, 5.9108, 6.1229, 5.9193, 6.0040, 6.0501, 6.0531, 6.1038, 6.0350, 6.1186, 6.1891, 6.1804, 6.1380, 6.0016, 5.9813,
        6.0483, 6.1838, 6.2075]
test = np.array(test)
test1 = test*1.15
test1[0] = 0.2149

x = np.arange(0,51+1)
optimization = np.full_like(x,1)*7.0685
plt.plot(test, label = "test")
plt.plot(test1, label = "test1")
plt.plot(x, optimization, label = "optimization")
plt.xlabel("number of epoch")
plt.ylabel('Reward')
plt.legend()
%matplotlib inline
plt.show()