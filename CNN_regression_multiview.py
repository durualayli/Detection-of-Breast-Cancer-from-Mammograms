import numpy as np
import os
import cv2
from keras.models import Model
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Input, Concatenate
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras import optimizers
import matplotlib.pyplot as plt

mlo_ben = []
path = os.listdir("/Users/ahmetalayli/Desktop/DL/MLO_full/MLO-ben-full-images")
for r in path:
    base, extension = os.path.splitext(r)
    base_without = base.replace("_MLO", "")
    new_path = base_without + extension
    mlo_ben.append(new_path)

mlo_ben_wout = []
path = os.listdir("/Users/ahmetalayli/Desktop/DL/MLO_full/MLO-ben-wout-full-images")
for r in path:
    base, extension = os.path.splitext(r)
    base_without = base.replace("_MLO", "")
    new_path = base_without + extension
    mlo_ben_wout.append(new_path)

mlo_mal = []
path = os.listdir("/Users/ahmetalayli/Desktop/DL/MLO_full/MLO-mal-full-images")
for r in path:
    base, extension = os.path.splitext(r)
    base_without = base.replace("_MLO", "")
    new_path = base_without + extension
    mlo_mal.append(new_path)

mlo_ben_test = []
path = os.listdir("/Users/ahmetalayli/Desktop/DL/MLO_full_test/MLO-ben-full-test-images")
for r in path:
    base, extension = os.path.splitext(r)
    base_without = base.replace("_MLO", "")
    new_path = base_without + extension
    mlo_ben_test.append(new_path)

mlo_ben_wout_test = []
path = os.listdir("/Users/ahmetalayli/Desktop/DL/MLO_full_test/MLO-ben-wout-full-test-images")
for r in path:
    base, extension = os.path.splitext(r)
    base_without = base.replace("_MLO", "")
    new_path = base_without + extension
    mlo_ben_wout_test.append(new_path)

mlo_mal_test = []
path = os.listdir("/Users/ahmetalayli/Desktop/DL/MLO_full_test/MLO-mal-full-test-images")
for r in path:
    base, extension = os.path.splitext(r)
    base_without = base.replace("_MLO", "")
    new_path = base_without + extension
    mlo_mal_test.append(new_path)

CC_ben = []
MLO_ben = []
path = os.listdir("/Users/ahmetalayli/Desktop/DL/CC_full/CC-ben-full-images")
for r in path:
    base, extension = os.path.splitext(r)
    base_without = base.replace("_CC", "")
    new_path = base_without + extension
    if new_path in mlo_ben:
        CC_ben.append(os.path.join("/Users/ahmetalayli/Desktop/DL/CC_full/CC-ben-full-images/"+r))
        MLO_path = base, extension = os.path.splitext(r)
        base_withoutt = base.replace("_CC", "_MLO")
        newest_path = base_withoutt + extension
        MLO_ben.append(os.path.join("/Users/ahmetalayli/Desktop/DL/MLO_full/MLO-ben-full-images/"+newest_path))

CC_ben_wout = []
MLO_ben_wout = []
path = os.listdir("/Users/ahmetalayli/Desktop/DL/CC_full/CC-ben-wout-full-images")
for r in path:
    base, extension = os.path.splitext(r)
    base_without = base.replace("_CC", "")
    new_path = base_without + extension
    if new_path in mlo_ben_wout:
        CC_ben_wout.append(os.path.join("/Users/ahmetalayli/Desktop/DL/CC_full/CC-ben-wout-full-images/"+r))
        MLO_path = base, extension = os.path.splitext(r)
        base_withoutt = base.replace("_CC", "_MLO")
        newest_path = base_withoutt + extension
        MLO_ben_wout.append(os.path.join("/Users/ahmetalayli/Desktop/DL/MLO_full/MLO-ben-wout-full-images/"+newest_path))

CC_mal = []
MLO_mal = []
path = os.listdir("/Users/ahmetalayli/Desktop/DL/CC_full/CC-mal-full-images")
for r in path:
    base, extension = os.path.splitext(r)
    base_without = base.replace("_CC", "")
    new_path = base_without + extension
    if new_path in mlo_mal:
        CC_mal.append(os.path.join("/Users/ahmetalayli/Desktop/DL/CC_full/CC-mal-full-images/"+r))
        MLO_path = base, extension = os.path.splitext(r)
        base_withoutt = base.replace("_CC", "_MLO")
        newest_path = base_withoutt + extension
        MLO_mal.append(os.path.join("/Users/ahmetalayli/Desktop/DL/MLO_full/MLO-mal-full-images/"+newest_path))

CC_ben_test = []
MLO_ben_test = []
path = os.listdir("/Users/ahmetalayli/Desktop/DL/CC_full_test/CC-ben-full-test-images")
for r in path:
    base, extension = os.path.splitext(r)
    base_without = base.replace("_CC", "")
    new_path = base_without + extension
    if new_path in mlo_ben_test:
        CC_ben_test.append(os.path.join("/Users/ahmetalayli/Desktop/DL/CC_full_test/CC-ben-full-test-images/"+r))
        MLO_path = base, extension = os.path.splitext(r)
        base_withoutt = base.replace("_CC", "_MLO")
        newest_path = base_withoutt + extension
        MLO_ben_test.append(os.path.join("/Users/ahmetalayli/Desktop/DL/MLO_full_test/MLO-ben-full-test-images/"+newest_path))

CC_ben_wout_test = []
MLO_ben_wout_test = []
path = os.listdir("/Users/ahmetalayli/Desktop/DL/CC_full_test/CC-ben-wout-full-test-images")
for r in path:
    base, extension = os.path.splitext(r)
    base_without = base.replace("_CC", "")
    new_path = base_without + extension
    if new_path in mlo_ben_wout_test:
        CC_ben_wout_test.append(os.path.join("/Users/ahmetalayli/Desktop/DL/CC_full_test/CC-ben-wout-full-test-images/"+r))
        MLO_path = base, extension = os.path.splitext(r)
        base_withoutt = base.replace("_CC", "_MLO")
        newest_path = base_withoutt + extension
        MLO_ben_wout_test.append(os.path.join("/Users/ahmetalayli/Desktop/DL/MLO_full_test/MLO-ben-wout-full-test-images/"+newest_path))

CC_mal_test = []
MLO_mal_test = []
path = os.listdir("/Users/ahmetalayli/Desktop/DL/CC_full_test/CC-mal-full-test-images")
for r in path:
    base, extension = os.path.splitext(r)
    base_without = base.replace("_CC", "")
    new_path = base_without + extension
    if new_path in mlo_mal_test:
        CC_mal_test.append(os.path.join("/Users/ahmetalayli/Desktop/DL/CC_full_test/CC-mal-full-test-images/"+r))
        MLO_path = base, extension = os.path.splitext(r)
        base_withoutt = base.replace("_CC", "_MLO")
        newest_path = base_withoutt + extension
        MLO_mal_test.append(os.path.join("/Users/ahmetalayli/Desktop/DL/MLO_full_test/MLO-mal-full-test-images/"+newest_path))

del (mlo_ben)
del (mlo_ben_wout)
del (mlo_mal)
del (mlo_ben_test)
del (mlo_ben_wout_test)
del (mlo_mal_test)

train_CC = []
test_CC = []

for path in CC_ben:
    img = cv2.imread(str(path))
    img = cv2.resize(img,(300,300), interpolation = cv2.INTER_LINEAR)
    img = img.astype("float32")
    img = img / 255
    train_CC.append([img,0.66])

for path in CC_ben_wout:
    img = cv2.imread(str(path))
    img = cv2.resize(img,(300,300), interpolation = cv2.INTER_LINEAR)
    img = img.astype("float32")
    img = img / 255
    train_CC.append([img,0.33])

for path in CC_mal:
    img = cv2.imread(str(path))
    img = cv2.resize(img,(300,300), interpolation = cv2.INTER_LINEAR)
    img = img.astype("float32")
    img = img / 255
    train_CC.append([img,1])

del (CC_ben)
del (CC_ben_wout)
del (CC_mal)

for path in CC_ben_test:
    img = cv2.imread(str(path))
    img = cv2.resize(img,(300,300), interpolation = cv2.INTER_LINEAR)
    img = img.astype("float32")
    img = img / 255
    test_CC.append([img,0.66])

for path in CC_ben_wout_test:
    img = cv2.imread(str(path))
    img = cv2.resize(img,(300,300), interpolation = cv2.INTER_LINEAR)
    img = img.astype("float32")
    img = img / 255
    test_CC.append([img,0.33])

for path in CC_mal_test:
    img = cv2.imread(str(path))
    img = cv2.resize(img,(300,300), interpolation = cv2.INTER_LINEAR)
    img = img.astype("float32")
    img = img / 255
    test_CC.append([img,1])

del (CC_ben_test)
del (CC_ben_wout_test)
del (CC_mal_test)

train_MLO = []
test_MLO = []

for path in MLO_ben:
    img = cv2.imread(str(path))
    img = cv2.resize(img,(300,300), interpolation = cv2.INTER_LINEAR)
    img = img.astype("float32")
    img = img / 255
    train_MLO.append([img,0.66])

for path in MLO_ben_wout:
    img = cv2.imread(str(path))
    img = cv2.resize(img,(300,300), interpolation = cv2.INTER_LINEAR)
    img = img.astype("float32")
    img = img / 255
    train_MLO.append([img,0.33])

for path in MLO_mal:
    img = cv2.imread(str(path))
    img = cv2.resize(img,(300,300), interpolation = cv2.INTER_LINEAR)
    img = img.astype("float32")
    img = img / 255
    train_MLO.append([img,1])

del (MLO_ben)
del (MLO_ben_wout)
del (MLO_mal)

for path in MLO_ben_test:
    img = cv2.imread(str(path))
    img = cv2.resize(img,(300,300), interpolation = cv2.INTER_LINEAR)
    img = img.astype("float32")
    img = img / 255
    test_MLO.append([img,0.66])

for path in MLO_ben_wout_test:
    img = cv2.imread(str(path))
    img = cv2.resize(img,(300,300), interpolation = cv2.INTER_LINEAR)
    img = img.astype("float32")
    img = img / 255
    test_MLO.append([img,0.33])

for path in MLO_mal_test:
    img = cv2.imread(str(path))
    img = cv2.resize(img,(300,300), interpolation = cv2.INTER_LINEAR)
    img = img.astype("float32")
    img = img / 255
    test_MLO.append([img,1])

del (MLO_ben_test)
del (MLO_ben_wout_test)
del (MLO_mal_test)

x_train_MLO = []
y_train_MLO = []

for img, risk in train_MLO:
    x_train_MLO.append(img)
    y_train_MLO.append(risk)

x_test_MLO = []
y_test_MLO = []

for img, risk in test_MLO:
    x_test_MLO.append(img)
    y_test_MLO.append(risk)

x_train_MLO = np.array(x_train_MLO)
y_train_MLO = np.array(y_train_MLO)

x_test_MLO = np.array(x_test_MLO)
y_test_MLO = np.array(y_test_MLO)

x_train_CC = []
y_train_CC = []

for img, risk in train_CC:
    x_train_CC.append(img)
    y_train_CC.append(risk)

x_test_CC = []
y_test_CC = []

for img, risk in test_CC:
    x_test_CC.append(img)
    y_test_CC.append(risk)

x_train_CC = np.array(x_train_CC)
y_train_CC = np.array(y_train_CC)

x_test_CC = np.array(x_test_CC)
y_test_CC = np.array(y_test_CC)

def soft_acc(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

optimizer = optimizers.legacy.Adam(learning_rate=0.001)

image_input = Input(shape=(300, 300, 3))
conv1 = Conv2D(filters = 32, kernel_size=(3, 3), activation='relu')(image_input)
pool1 = MaxPool2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(filters = 32, kernel_size=(5, 5), activation='relu')(pool1)
pool2 = MaxPool2D(pool_size=(2, 2))(conv2)
conv3 = Conv2D(filters = 64, kernel_size=(7, 7), activation='relu')(pool2)
pool3 = MaxPool2D(pool_size=(2, 2))(conv3)
conv4 = Conv2D(filters = 64, kernel_size=(9, 9), activation='relu')(pool3)
pool4 = MaxPool2D(pool_size=(2, 2))(conv4)
flatten1 = Flatten()(pool4)

image_input2 = Input(shape=(300, 300, 3))
conv5 = Conv2D(filters = 32, kernel_size=(3, 3), activation='relu')(image_input2)
pool5 = MaxPool2D(pool_size=(2, 2))(conv5)
conv6 = Conv2D(filters = 32, kernel_size=(5, 5), activation='relu')(pool5)
pool6 = MaxPool2D(pool_size=(2, 2))(conv6)
conv7 = Conv2D(filters = 64, kernel_size=(7, 7), activation='relu')(pool6)
pool7 = MaxPool2D(pool_size=(2, 2))(conv7)
conv8 = Conv2D(filters = 64, kernel_size=(9, 9), activation='relu')(pool7)
pool8 = MaxPool2D(pool_size=(2, 2))(conv8)
flatten2 = Flatten()(pool8)

concatenated = Concatenate()([flatten1, flatten2])
dense1 = Dense(128, activation = "relu")(concatenated)
dense2 = Dense(64, activation = "relu")(dense1)
model = Model(inputs=[image_input, image_input2], outputs=Dense(1, activation="linear")(concatenated))
print("x_train_MLO shape:", x_train_MLO.shape)
print("x_train_CC shape:", x_train_CC.shape)
print("y_train_MLO shape:", y_train_MLO.shape)
print("y_train_CC shape:", y_train_CC.shape)

print("x_test_MLO shape:", x_test_MLO.shape)
print("x_test_CC shape:", x_test_CC.shape)
print("y_test_MLO shape:", y_test_MLO.shape)
print("y_test_CC shape:", y_test_CC.shape)
model.compile(loss='mse', optimizer=optimizer,metrics=[soft_acc])
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit([x_train_MLO, x_train_CC], [y_train_MLO,y_train_CC], epochs = 30, validation_data = ([x_test_MLO, x_test_CC],[y_test_MLO, y_test_CC]),callbacks=[early_stopping])

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