import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense
from keras.models import Model

is_init = False
size = -1
label= [] #for handling the y values 
dict = {} #converting the string to integer values
c = 0
for i in os.listdir():
    if i.split(".")[-1] == "npy" and not(i.split(".")[0] == "lables"): #to separate the .npy files in the directory and loading it
        if not(is_init):
            is_init = True
            x = np.load(i) #this loads the data inside the file 
            size = x.shape[0]
            y = np.array([i.split(".")[0]]*size).reshape(-1, 1)#this loads the name of the file for inference
        else:
            x = np.concatenate((x, np.load(i)))
            y = np.concatenate((y, np.array([i.split(".")[0]]*size).reshape(-1, 1)))

        label.append(i.split(".")[0])
        dict[i.split(".")[0]] = c
        c = c + 1

#to convert the y to integer
for i in range(y.shape[0]):
    y[i, 0] = dict[y[i, 0]]
y = np.array(y, dtype="int32")
print(y.shape)

#if good is 1 and bad is 2 we cannot directly compare them and say then good si below bad or this is the precedence of the label instead
#we use tensorflow library to handle this.

y = to_categorical(y) #this basically convets it to vectors

#shuffling data to increase its efficiency
x_new = x.copy()
y_new = y.copy()
counter = 0

cnt = np.arange(x.shape[0])
np.random.shuffle(cnt) #this shuffles all the values

for i in cnt:
    x_new[counter] = x[i]
    y_new[counter] = y[i]
    counter = counter + 1

#print(x.shape)
print(y.shape)

ip = Input(shape=(x.shape[1],))

m = Dense(512, activation = "relu")(ip)
m = Dense(256, activation = "relu")(m)

op = Dense(y.shape[1], activation = "softmax")(m)
model = Model(inputs = ip, outputs = op)
model.compile(optimizer = "rmsprop", loss = "categorical_crossentropy", metrics=["acc"])
model.fit(x, y, epochs = 50)

model.save("model.h5")
np.save("lables.npy", np.array(label))