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
#     init_model = ResNet50(include_top = False, classes = 10, input_shape = (32, 32, 3))
    input_ = Input(shape = IMG_SHAPE)
    conv_7x7_2 = Conv2D(64, kernel_size = (7, 7), strides = 2, activation = 'relu')(input_)
    re_0 = Activation('relu')(conv_7x7_2)
    bn_1 = BatchNormalization()(re_0)
    maxpool_7x7_2 = MaxPool2D(pool_size = (2, 2))(bn_1)

    conv_3x3_1_a = Conv2D(64, kernel_size = (3, 3), activation = 'relu', padding = 'same')(maxpool_7x7_2)
    re_1 = Activation('relu')(conv_3x3_1_a)
    bn_2 = BatchNormalization()(re_1)
    conv_3x3_1_b = Conv2D(64, kernel_size = (3, 3), activation = 'relu', padding = 'same')(bn_2)
    add_1 = Add()([conv_3x3_1_b, maxpool_7x7_2])

    conv_3x3_1_c = Conv2D(64, kernel_size = (3, 3), activation = 'relu', padding = 'same')(add_1)
    re_2 = Activation('relu')(conv_3x3_1_c)
    bn_3 = BatchNormalization()(re_2)
    conv_3x3_1_d = Conv2D(64, kernel_size = (3, 3), activation = 'relu', padding = 'same')(bn_3)
    add_2 = Add()([conv_3x3_1_d, add_1])

    conv_3x3_1_e = Conv2D(64, kernel_size = (3, 3), activation = 'relu', padding = 'same')(add_2)
    re_3 = Activation('relu')(conv_3x3_1_e)
    bn_4 = BatchNormalization()(re_3)
    conv_3x3_1_f = Conv2D(64, kernel_size = (3, 3), activation = 'relu', padding = 'same')(bn_4)
    add_3 = Add()([conv_3x3_1_f, add_2])
    
    conv_3x3_2 = Conv2D(64, kernel_size = (3, 3), activation = 'relu', padding = 'same')
    
    glob_avg_pool = AveragePooling2D((3, 3))(add_3)
    flatten_ = Flatten()(glob_avg_pool)
    softmax = Dense(10, activation = 'softmax')(flatten_)
    return Model(input_
                 , softmax)


model = resnet_model()
model.summary()

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit_generator(datagen.flow(X_train, y_train, batch_size = 32), epochs = 100, steps_per_epoch = X_train.shape[0] / 32, validation_data = (X_val, y_val))

loss, acc = model.evaluate(X_test, y_test)
print('loss: {}, acc: {}'.format(loss, acc))


#Note1: train with all conv layer and 100 epochs: acc = .7798 while train-acc: .9941 => overfit
#Note2: train with two first block conv of resnet and 100 epochs: acc = .7295 while train-acc = .9864
#Note3: train with first block conv of restnet and 100 epochs: acc = .7058 while train-acc = .9808
#Note4: augmentation data and train with 25 epochs: acc = .777 
#Note5: augmentation data and train with 50 epochs, model with two first conv block: acc = .7988 val_acc: 0.8088
#Note6: same model with note 5, train with 100 epochs: acc = .827 val_acc = .83
