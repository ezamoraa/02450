from data import *
from matplotlib.pyplot import *

Attributes = range(0, len(attributeNames))
# Attributes = [0, 1, 2, 3]
# Attributes = [4, 6]
NumAtr = len(Attributes)

figure(figsize=(12, 12))
for m1 in range(NumAtr):
    for m2 in range(NumAtr):
        subplot(NumAtr, NumAtr, m1 * NumAtr + m2 + 1)
        for c in range(C):
            class_mask = y == c
            plot(X[class_mask, Attributes[m2]], X[class_mask, Attributes[m1]], ".")
            if m1 == NumAtr - 1:
                xlabel(attributeNames[Attributes[m2]])
            else:
                xticks([])
            if m2 == 0:
                ylabel(attributeNames[Attributes[m1]])
            else:
                yticks([])
legend(classNames)
show()
