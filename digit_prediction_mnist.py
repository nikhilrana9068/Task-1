from tensorflow.keras.datasets import mnist,cifar10
import numpy as np
(train_X, train_y), (test_X, test_y)=mnist.load_data()
#(train_X, train_y), (test_X, test_y)=cifar10.load_data()
# print(type(train_X))

print(train_X.shape)

# train_X=train_X.astype(np.uint8)

train_X=train_X.reshape(train_X.shape[0],train_X.shape[1],train_X.shape[2],1)
test_X=test_X.reshape(test_X.shape[0],test_X.shape[1],test_X.shape[2],1)
print(train_X.shape)

train_X=train_X.astype("float32")
test_X=test_X.astype("float32")

train_X=train_X/255
test_X=test_X/255

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential

model=Sequential()
in_shape=train_X.shape[1:]
model.add(Conv2D(2,(2,2),activation="relu",input_shape=in_shape))
model.add(MaxPool2D((3,3)))
model.add(Conv2D(2,(2,2),activation="relu",input_shape=in_shape))
model.add(MaxPool2D((3,3)))

model.add(Flatten())

model.add(Dense(100,activation="relu",kernel_initializer="he_normal"))
model.add(Dense(60,activation="relu",kernel_initializer="he_normal"))
model.add(Dense(10,activation="sigmoid"))


model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=['accuracy'])

model.fit(train_X,train_y,batch_size=32,epochs=50,verbose=0)

loss,acc=model.evaluate(test_X,test_y)

# model.save('cnn_model.h5')


print("Accuracy:",acc)

y_pred=model.predict(np.asarray(test_X[0]))


print(y_pred)
print(y_test[0])





print(train_X.shape)

print(test_X.shape)
print(train_y.shape)
print(np.unique(train_y))

import matplotlib.pyplot as plt

for x in range(25):
  plt.subplot(5,5,x+1)
  plt.imshow(train_X[x])
  print(train_y[x])
plt.show()