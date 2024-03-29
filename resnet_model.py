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
    conv_7x7_0 = Conv2D(64, kernel_size = (7, 7), strides = 2)(input_)
    bn_0_a = BatchNormalization()(conv_7x7_0)
    relu_0_a = Activation('relu')(bn_0_a)
    maxpool_7x7_0 = MaxPool2D(pool_size = (2, 2))(relu_0_a)

    conv_3x3_1_a = Conv2D(64, kernel_size = (3, 3), padding = 'same')(maxpool_7x7_0)
    bn_1_a = BatchNormalization()(conv_3x3_1_a)
    relu_1_a = Activation('relu')(bn_1_a)
    conv_3x3_1_b = Conv2D(64, kernel_size = (3, 3), padding = 'same')(relu_1_a)
    bn_1_b = BatchNormalization()(conv_3x3_1_b)
    relu_1_b = Activation('relu')(bn_1_b)
    add_1 = Add()([relu_1_b, maxpool_7x7_0])

    conv_3x3_1_c = Conv2D(64, kernel_size = (3, 3), padding = 'same')(add_1)
    bn_1_c = BatchNormalization()(conv_3x3_1_c)
    relu_1_c = Activation('relu')(bn_1_c)
    conv_3x3_1_d = Conv2D(64, kernel_size = (3, 3), padding = 'same')(relu_1_c)
    bn_1_d = BatchNormalization()(conv_3x3_1_d)
    relu_1_d = Activation('relu')(bn_1_d)
    add_2 = Add()([add_1, relu_1_d])

    conv_3x3_1_e = Conv2D(64, kernel_size = (3, 3), padding = 'same')(add_2)
    bn_1_e = BatchNormalization()(conv_3x3_1_e)
    relu_1_e = Activation('relu')(bn_1_e)
    conv_3x3_1_f = Conv2D(64, kernel_size = (3, 3), padding = 'same')(relu_1_e)
    bn_1_f = BatchNormalization()(conv_3x3_1_f)
    relu_1_f = Activation('relu')(bn_1_f)
    add_3 = Add()([add_2, relu_1_f])
    conv_1x1_1 = Conv2D(128, kernel_size = (1, 1), strides = 2, padding = 'same')(add_3)

    conv_3x3_2 = Conv2D(128, kernel_size = (3, 3), strides = 2, padding = 'same')(add_3)
    bn_2 = BatchNormalization()(conv_3x3_2)
    relu_2 = Activation('relu')(bn_2)
    conv_3x3_2_a = Conv2D(128, kernel_size = (3, 3), padding = 'same')(relu_2)
    bn_2_a = BatchNormalization()(conv_3x3_2_a)
    relu_2_a = Activation('relu')(bn_2_a)
    add_1_ = Add()([relu_2_a, conv_1x1_1])

    conv_3x3_2_b = Conv2D(128, kernel_size = (3, 3), padding = 'same')(add_1_)
    bn_2_b = BatchNormalization()(conv_3x3_2_b)
    relu_2_b = Activation('relu')(bn_2_b)
    conv_3x3_2_c = Conv2D(128, kernel_size = (3, 3), padding = 'same')(relu_2_b)
    bn_2_c = BatchNormalization()(conv_3x3_2_c)
    relu_2_c = Activation('relu')(bn_2_c)
    add_4 = Add()([add_1_, relu_2_c])

    conv_3x3_2_d = Conv2D(128, kernel_size = (3, 3), padding = 'same')(add_4)
    bn_2_d = BatchNormalization()(conv_3x3_2_d)
    relu_2_d = Activation('relu')(bn_2_d)
    conv_3x3_2_e = Conv2D(128, kernel_size = (3, 3), padding = 'same')(relu_2_d)
    bn_2_e = BatchNormalization()(conv_3x3_2_e)
    relu_2_e = Activation('relu')(bn_2_e)
    add_5 = Add()([relu_2_e, add_4])

    conv_3x3_2_f = Conv2D(128, (3, 3), padding = 'same')(add_5)
    bn_2_f = BatchNormalization()(conv_3x3_2_f)
    relu_2_f = Activation('relu')(bn_2_f)
    conv_3x3_2_g = Conv2D(128, (3, 3), padding = 'same')(relu_2_f)
    bn_2_g = BatchNormalization()(conv_3x3_2_g)
    relu_2_g = Activation('relu')(bn_2_g)
    add_6 = Add()([relu_2_g, add_5])
    conv_1x1_2 = Conv2D(256, kernel_size = (1, 1), padding = 'same', strides = 2)(add_6)

    conv_3x3_3 = Conv2D(256, (3, 3), strides = 2, padding = 'same')(add_6)
    bn_3 = BatchNormalization()(conv_3x3_3)
    relu_3 = Activation('relu')(bn_3)
    conv_3x3_3_a = Conv2D(256, (3, 3), padding = 'same')(relu_3)
    bn_3_a = BatchNormalization()(conv_3x3_3_a)
    relu_3_a = Activation('relu')(bn_3_a)
    add_2_ = Add()([relu_3_a, conv_1x1_2])

    conv_3x3_3_b = Conv2D(256, (3, 3), padding = 'same')(add_2_)
    bn_3_b = BatchNormalization()(conv_3x3_3_b)
    relu_3_b = Activation('relu')(bn_3_b)
    conv_3x3_3_c = Conv2D(256, (3, 3), padding = 'same')(relu_3_b)
    bn_3_c = BatchNormalization()(conv_3x3_3_c)
    relu_3_c = Activation('relu')(bn_3_c)
    add_7 = Add()([add_2_, relu_3_c])

    conv_3x3_3_d = Conv2D(256, (3, 3), padding = 'same')(add_7)
    bn_3_d = BatchNormalization()(conv_3x3_3_d)
    relu_3_d = Activation('relu')(bn_3_d)
    conv_3x3_3_e = Conv2D(256, (3, 3), padding = 'same')(relu_3_d)
    bn_3_e = BatchNormalization()(conv_3x3_3_e)
    relu_3_e = Activation('relu')(bn_3_e)
    add_8 = Add()([add_7, relu_3_e])
    
    conv_3x3_3_f = Conv2D(256, (3, 3), padding = 'same')(add_8)
    bn_3_f = BatchNormalization()(conv_3x3_3_f)
    relu_3_f = Activation('relu')(bn_3_f)
    conv_3x3_3_g = Conv2D(256, (3, 3), padding = 'same')(relu_3_f)
    bn_3_g = BatchNormalization()(conv_3x3_3_g)
    relu_3_g = Activation('relu')(bn_3_g)
    add_9 = Add()([add_8, relu_3_g])


    conv_3x3_3_h = Conv2D(256, (3, 3), padding = 'same')(add_9)
    bn_3_h = BatchNormalization()(conv_3x3_3_h)
    relu_3_h = Activation('relu')(bn_3_h)
    conv_3x3_3_i = Conv2D(256, (3, 3), padding = 'same')(relu_3_h)
    bn_3_i = BatchNormalization()(conv_3x3_3_i)
    relu_3_i = Activation('relu')(bn_3_i)
    add_10 = Add()([add_9, relu_3_i])
    conv_1x1_3 = Conv2D(512, kernel_size = (1, 1), padding = 'same', strides = 2)(add_10)

    conv_3x3_4 = Conv2D(512, (3, 3), strides = 2, padding = 'same')(add_10)
    bn_4 = BatchNormalization()(conv_3x3_4)
    relu_4 = Activation('relu')(bn_4)
    conv_3x3_4_a = Conv2D(512, (3, 3), padding = 'same')(relu_4)
    bn_4_a = BatchNormalization()(conv_3x3_4_a)
    relu_4_a = Activation('relu')(bn_4_a)
    add_3_ = Add()([relu_4_a, conv_1x1_3])

    conv_3x3_4_b = Conv2D(512, (3, 3), padding = 'same')(add_3_)
    bn_4_b = BatchNormalization()(conv_3x3_4_b)
    relu_4_b = Activation('relu')(bn_4_b)
    conv_3x3_4_c = Conv2D(512, (3, 3), padding = 'same')(relu_4_b)
    bn_4_c = BatchNormalization()(conv_3x3_4_c)
    relu_4_c = Activation('relu')(bn_4_c)
    add_11 = Add()([add_3_, relu_4_c])

    conv_3x3_4_d = Conv2D(512, (3, 3), padding = 'same')(add_11)
    bn_4_d = BatchNormalization()(conv_3x3_4_d)
    relu_4_d = Activation('relu')(bn_4_d)
    conv_3x3_4_e = Conv2D(512, (3, 3), padding = 'same')(relu_4_d)
    bn_4_e = BatchNormalization()(conv_3x3_4_e)
    relu_4_e = Activation('relu')(bn_4_e)
    add_12 = Add()([add_11, relu_4_e])

    conv_3x3_4_f = Conv2D(512, (3, 3), padding = 'same')(add_12)
    bn_4_f = BatchNormalization()(conv_3x3_4_f)
    relu_4_f = Activation('relu')(bn_4_f)
    conv_3x3_4_g = Conv2D(512, (3, 3), padding = 'same')(relu_4_f)
    bn_4_g = BatchNormalization()(conv_3x3_4_g)
    relu_4_g = Activation('relu')(bn_4_g)
    add_13 = Add()([add_12, relu_4_g])

    flatten_ = Flatten()(add_13)
    softmax = Dense(10, activation = 'softmax')(flatten_)

    return Model(input_, outputs = [softmax])

model = resnet_model()
model.summary()


model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit_generator(datagen.flow(X_train, y_train, batch_size = 32), epochs = 100, steps_per_epoch = X_train.shape[0] / 32, validation_data = (X_val, y_val))
loss, acc = model.evaluate(X_test, y_test)
print('loss: {}, acc: {}'.format(loss, acc))


#first train: acc: 0.7962
