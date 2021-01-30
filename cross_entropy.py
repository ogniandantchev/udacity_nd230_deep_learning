import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.


def cross_entropy(Y, P):
    Y = np.float_(Y)
    P = np.float_(P)
    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))



#def cross_entropy(Y, P):
#    return -np.sum([Y[i]* np.log(P[i]) for i in range(len(Y))])


# Trying for Y=[1,0,1,1] and P=[0.4,0.6,0.1,0.5].
# The correct answer is
# 4.8283137373
# And your code returned
# 4.8283137373

# Correct!
