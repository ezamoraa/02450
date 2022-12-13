import numpy as np

""" 
Try the script on:
2016 fall Q20
2018 fall Q22
2019 fall Q20
2021 fall Q18 (difficult setup)
 """

def sigmoid(xw):
    output = 1 / (1 + np.exp(-xw))
    return output

def reLu(xw):
    output = (xw > 0) * xw
    return output

def tanh(xw):
    output = np.tanh(xw)
    return output


""" Either sigmoid og reLu """
transferfunction = "sigmoid"
w0 = 1.4
w1 = [-0.5, -0.1]
w2 = [0.9, 2]
# w3 = [-0.1, 2.1]
w_out = np.array([-1, 0.4]).transpose()

w_list = np.array([w1, w2])

x = np.array([1, -2])
# x = np.array([
#     [1, 0, 0],
#     [1, 1, 0],
#     [1, 1, 1],
#     [1, 1, 2]
#     ])


h = []

if transferfunction == "sigmoid":
    hFunc = sigmoid
elif transferfunction == "reLu":
    hFunc = reLu
elif transferfunction == "tanh":
    hFunc = tanh
else:
    print("Pick a valid transferfunction")


for weight in w_list:
    xw = x @ weight
    h.append(hFunc(xw))

""" Use this one instead for somehting like 2021 fall Q18"""
# z = np.array( [np.ones_like(h), h] ).transpose().squeeze()
# f = np.sum(w_out * z, axis=1)+ w0

""" Use this one for 2019 fall Q20. It has an additional transferfunction on the output """
f = sigmoid(np.sum(w_out * h) + w0)

""" Use this for simple exercises like 2016 fall Q20 and 2018 fall Q22"""
# f = np.sum(w_out * h + w0)

print(f)
