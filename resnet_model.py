from keras.layers import *
from keras.models import *
from keras.applications.resnet50 import *
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = preprocess_input(X_train)
X_test = preprocess_input(X_test)
# X_test = X_test / 255
IMG_SHAPE = X_train.shape[1:]

endp = int(X_train.shape[0] * .9)
X_val = X_train[endp:]
y_val = y_train[endp:]
X_train = X_train[:endp]
y_train = y_train[:endp]

datagen = ImageDataGenerator(rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

def resnet_model():
    input_ = Input(shape = IMG_SHAPE)
    conv_7x7_2 = Conv2D(64, kernel_size = (7, 7), strides = 2, activation = 'relu')(input_)
    bn_1 = BatchNormalization()(conv_7x7_2)
    maxpool_7x7_2 = MaxPool2D(pool_size = (2, 2))(bn_1)

    conv_3x3_1_a = Conv2D(64, kernel_size = (3, 3), activation = 'relu', padding = 'same')(maxpool_7x7_2)
    bn_2 = BatchNormalization()(conv_3x3_1_a)
    conv_3x3_1_b = Conv2D(64, kernel_size = (3, 3), activation = 'relu', padding = 'same')(bn_2)
    add_1 = Add()([conv_3x3_1_b, maxpool_7x7_2])

    conv_3x3_1_c = Conv2D(64, kernel_size = (3, 3), activation = 'relu', padding = 'same')(add_1)
    bn_3 = BatchNormalization()(conv_3x3_1_c)
    conv_3x3_1_d = Conv2D(64, kernel_size = (3, 3), activation = 'relu', padding = 'same')(bn_3)
    add_2 = Add()([conv_3x3_1_d, add_1])

    conv_3x3_1_e = Conv2D(64, kernel_size = (3, 3), activation = 'relu', padding = 'same')(add_2)
    bn_4 = BatchNormalization()(conv_3x3_1_e)
    conv_3x3_1_f = Conv2D(64, kernel_size = (3, 3), activation = 'relu', padding = 'same')(bn_4)
    add_3 = Add()([conv_3x3_1_f, add_2])
    
    conv_3x3_2 = Conv2D(128, kernel_size = (3, 3), strides = 2, activation = 'relu', padding = 'same')(add_3)
    bn_5 = BatchNormalization()(conv_3x3_2)
    conv_3x3_2_a = Conv2D(128, kernel_size = (3, 3), activation = 'relu', padding = 'same')(bn_5)
    bn_6 = BatchNormalization()(conv_3x3_2_a)
    conv_3x3_2_b = Conv2D(128, kernel_size = (3, 3), activation = 'relu', padding = 'same')(bn_6)
    bn_7 = BatchNormalization()(conv_3x3_2_b)
    conv_3x3_2_c = Conv2D(128, kernel_size = (3, 3), activation = 'relu', padding = 'same')(bn_7)
    add_4 = Add()([conv_3x3_2_c, bn_6])

    conv_3x3_2_d = Conv2D(128, kernel_size = (3, 3), activation = 'relu', padding = 'same')(add_4)
    bn_8 = BatchNormalization()(conv_3x3_2_d)
    conv_3x3_2_e = Conv2D(128, kernel_size = (3, 3), activation = 'relu', padding = 'same')(bn_8)
    bn_9 = BatchNormalization()(conv_3x3_2_e)
    add_5 = Add()([bn_9, add_4])

    conv_3x3_2_f = Conv2D(128, (3, 3), padding = 'same', activation = 'relu')(add_5)
    bn_10 = BatchNormalization()(conv_3x3_2_f)
    conv_3x3_2_g = Conv2D(128, (3, 3), padding = 'same', activation = 'relu')(bn_10)
    bn_11 = BatchNormalization()(conv_3x3_2_g)
    add_6 = Add()([bn_11, conv_3x3_2_f])

    conv_3x3_3 = Conv2D(256, (3, 3), strides = 2, padding = 'same', activation = 'relu')(add_6)
    bn_12 = BatchNormalization()(conv_3x3_3)
    conv_3x3_3_a = Conv2D(256, (3, 3), padding = 'same', activation = 'relu')(bn_12)
    bn_13 = BatchNormalization()(conv_3x3_3_a)
    conv_3x3_3_b = Conv2D(256, (3, 3), padding = 'same', activation = 'relu')(bn_13)
    bn_14 = BatchNormalization()(conv_3x3_3_b)
    conv_3x3_3_c = Conv2D(256, (3, 3), padding = 'same', activation = 'relu')(bn_14)
    bn_15 = BatchNormalization()(conv_3x3_3_c)
    add_7 = Add()([bn_13, bn_15])

    conv_3x3_3_d = Conv2D(256, (3, 3), padding = 'same', activation = 'relu')(add_7)
    bn_16 = BatchNormalization()(conv_3x3_3_d)
    conv_3x3_3_e = Conv2D(256, (3, 3), padding = 'same', activation = 'relu')(bn_16)
    bn_17 = BatchNormalization()(conv_3x3_3_e)
    add_8 = Add()([add_7, bn_17])
    
    conv_3x3_3_f = Conv2D(256, (3, 3), padding = 'same', activation = 'relu')(add_8)
    bn_18 = BatchNormalization()(conv_3x3_3_f)
    conv_3x3_3_g = Conv2D(256, (3, 3), padding = 'same', activation = 'relu')(bn_18)
    bn_19 = BatchNormalization()(conv_3x3_3_g)
    add_9 = Add()([add_8, bn_19])


    conv_3x3_3_h = Conv2D(256, (3, 3), padding = 'same', activation = 'relu')(add_9)
    bn_20 = BatchNormalization()(conv_3x3_3_h)
    conv_3x3_3_i = Conv2D(256, (3, 3), padding = 'same', activation = 'relu')(bn_20)
    bn_21 = BatchNormalization()(conv_3x3_3_i)
    add_10 = Add()([add_9, bn_21])

    conv_3x3_4 = Conv2D(512, (3, 3), strides = 2, padding = 'same', activation = 'relu')(add_10)
    bn_22 = BatchNormalization()(conv_3x3_4)
    conv_3x3_4_a = Conv2D(512, (3, 3), padding = 'same', activation = 'relu')(bn_22)
    bn_23 = BatchNormalization()(conv_3x3_4_a)
    conv_3x3_4_b = Conv2D(512, (3, 3), padding = 'same', activation = 'relu')(bn_23)
    bn_24 = BatchNormalization()(conv_3x3_4_b)
    conv_3x3_4_c = Conv2D(512, (3, 3), padding = 'same', activation = 'relu')(bn_24)
    add_11 = Add()([conv_3x3_4_a, conv_3x3_4_c])

    conv_3x3_4_d = Conv2D(512, (3, 3), padding = 'same', activation = 'relu')(add_11)
    bn_25 = BatchNormalization()(conv_3x3_4_d)
    conv_3x3_4_e = Conv2D(512, (3, 3), padding = 'same', activation = 'relu')(bn_25)
    bn_26 = BatchNormalization()(conv_3x3_4_e)
    add_12 = Add()([add_11, bn_26])

    conv_3x3_4_f = Conv2D(512, (3, 3), padding = 'same', activation = 'relu')(add_12)
    bn_27 = BatchNormalization()(conv_3x3_4_f)
    conv_3x3_4_g = Conv2D(512, (3, 3), padding = 'same', activation = 'relu')(bn_27)
    bn_28 = BatchNormalization()(conv_3x3_4_g)
    add_13 = Add()([add_12, bn_28])

    flatten_ = Flatten()(add_13)
    softmax = Dense(10, activation = 'softmax')(flatten_)

    return Model(input_, outputs = [softmax])

model = resnet_model()
model.summary()

model.fit_generator(datagen.flow(X_train, y_train, batch_size = 32), epochs = 100, steps_per_epoch = X_train.shape[0] / 32, validation_data = (X_val, y_val))
model.fit(X_train, y_train, epochs = 100, batch_size = 32, validation_split = 0.1)

loss, acc = model.evaluate(X_test, y_test)
print('loss: {}, acc: {}'.format(loss, acc))


# model.summary()
