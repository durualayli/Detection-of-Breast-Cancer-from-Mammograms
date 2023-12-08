import numpy as np
import random
import os
import cv2
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, InputLayer
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras import optimizers
import matplotlib.pyplot as plt

MLO_ben = [os.path.join("/Users/ahmetalayli/Desktop/DL/MLO_full/MLO-ben-full-images/"+r) for r in os.listdir("/Users/ahmetalayli/Desktop/DL/MLO_full/MLO-ben-full-images")]
MLO_ben_wout = [os.path.join("/Users/ahmetalayli/Desktop/DL/MLO_full/MLO-ben-wout-full-images/"+r) for r in os.listdir("/Users/ahmetalayli/Desktop/DL/MLO_full/MLO-ben-wout-full-images")]
MLO_mal = [os.path.join("/Users/ahmetalayli/Desktop/DL/MLO_full/MLO-mal-full-images/"+r) for r in os.listdir("/Users/ahmetalayli/Desktop/DL/MLO_full/MLO-mal-full-images")]

train = []
test = []

ben = 0
ben_wout = 0
mal = 0

for path in MLO_ben:
    img = cv2.imread(str(path))
    img = cv2.resize(img,(300,300), interpolation = cv2.INTER_LINEAR)
    img = img.astype("float32")
    img = img / 255
    train.append([img,0.66])
    ben = ben + 1

for path in MLO_ben_wout:
    img = cv2.imread(str(path))
    img = cv2.resize(img,(300,300), interpolation = cv2.INTER_LINEAR)
    img = img.astype("float32")
    img = img / 255
    train.append([img,0.33])
    ben_wout = ben_wout + 1

for path in MLO_mal:
    img = cv2.imread(str(path))
    img = cv2.resize(img,(300,300), interpolation = cv2.INTER_LINEAR)
    img = img.astype("float32")
    img = img / 255
    train.append([img,1])
    mal = mal + 1

print (ben)
print (ben_wout)
print (mal)

del (MLO_ben)
del (MLO_ben_wout)
del (MLO_mal)

MLO_ben_test = [os.path.join("/Users/ahmetalayli/Desktop/DL/MLO_full_test/MLO-ben-full-test-images/"+r) for r in os.listdir("/Users/ahmetalayli/Desktop/DL/MLO_full_test/MLO-ben-full-test-images")]
MLO_ben_wout_test = [os.path.join("/Users/ahmetalayli/Desktop/DL/MLO_full_test/MLO-ben-wout-full-test-images/"+r) for r in os.listdir("/Users/ahmetalayli/Desktop/DL/MLO_full_test/MLO-ben-wout-full-test-images")]
MLO_mal_test = [os.path.join("/Users/ahmetalayli/Desktop/DL/MLO_full_test/MLO-mal-full-test-images/"+r) for r in os.listdir("/Users/ahmetalayli/Desktop/DL/MLO_full_test/MLO-mal-full-test-images")]

for path in MLO_ben_test:
    img = cv2.imread(str(path))
    img = cv2.resize(img,(300,300), interpolation = cv2.INTER_LINEAR)
    img = img.astype("float32")
    img = img / 255
    test.append([img,0.66])

for path in MLO_ben_wout_test:
    img = cv2.imread(str(path))
    img = cv2.resize(img,(300,300), interpolation = cv2.INTER_LINEAR)
    img = img.astype("float32")
    img = img / 255
    test.append([img,0.33])

for path in MLO_mal_test:
    img = cv2.imread(str(path))
    img = cv2.resize(img,(300,300), interpolation = cv2.INTER_LINEAR)
    img = img.astype("float32")
    img = img / 255
    test.append([img,1])

del (MLO_ben_test)
del (MLO_ben_wout_test)
del (MLO_mal_test)

random.shuffle(train)
random.shuffle(test)

x_train = []
y_train = []

for img, risk in train:
    x_train.append(img)
    y_train.append(risk)

x_test = []
y_test = []

for img, risk in test:
    x_test.append(img)
    y_test.append(risk)

x_train = np.array(x_train)
y_train = np.array(y_train)

x_test = np.array(x_test)
y_test = np.array(y_test)

model = Sequential()
model.add(InputLayer(input_shape=(300, 300, 3)))
model.add(Conv2D(filters = 32, kernel_size = (3,3), activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters = 32, kernel_size = (5,5), activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters = 64, kernel_size = (7,7), activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters = 64, kernel_size = (9,9), activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128, activation = "relu"))
model.add(Dense(64, activation = "relu"))
model.add(Dense(1, activation = "linear"))

def soft_acc(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))
optimizer = optimizers.legacy.Adam(learning_rate=0.001)
model.compile(loss='mse', optimizer=optimizer,metrics=[soft_acc])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(x_train, y_train, epochs = 30, validation_data = (x_test,y_test), callbacks=[early_stopping])

plt.plot(history.history['soft_acc'])
plt.plot(history.history['val_soft_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()