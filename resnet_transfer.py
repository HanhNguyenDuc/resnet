from keras.layers import *
from keras.models import *
from keras.applications.resnet50 import *
from keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = preprocess_input(X_train)
X_test = preprocess_input(X_test)
# X_test = X_test / 255
IMG_SHAPE = X_train.shape[1:]

def resnet_model():
    init_model = ResNet50(include_top = False, classes = 10, input_shape = (32, 32, 3))
    
    # x = init_model.input
    # print(x)
    
    # for i, layer in enumerate(init_model.layers[1:], 1):
    #   if i < 29:
    #     x = layer(x)
    
#     glob_avg_pool = AveragePooling2D((2, 2))(x)
    flatten_ = Flatten()(init_model.output)
    softmax = Dense(10, activation = 'softmax')(flatten_)
    return Model(init_model.input, softmax)


def resnet_model_2():
#     init_model = ResNet50(include_top = False, classes = 10, input_shape = (32, 32, 3))
    input_ = Input(shape = IMG_SHAPE)
    conv_7x7_2 = Conv2D(64, kernel_size = (7, 7), strides = 2, activation = 'relu')(input_)
    bn_1 = BatchNormalization()(conv_7x7_2)
    maxpool_7x7_2 = MaxPool2D(pool_size = (2, 2))(bn_1)

    conv_3x3_1_a = Conv2D(64, kernel_size = (3, 3), activation = 'relu', padding = 'same')(maxpool_7x7_2)
    bn_2 = BatchNormalization()(conv_3x3_1_a)
    conv_3x3_1_b = Conv2D(64, kernel_size = (3, 3), activation = 'relu', padding = 'same')(bn_2)
    add_1 = Add()([conv_3x3_1_b, maxpool_7x7_2])

#     conv_3x3_1_c = Conv2D(64, kernel_size = (3, 3), activation = 'relu', padding = 'same')(add_1)
#     bn_3 = BatchNormalization()(conv_3x3_1_c)
#     conv_3x3_1_d = Conv2D(64, kernel_size = (3, 3), activation = 'relu', padding = 'same')(bn_3)
#     add_2 = Add()([conv_3x3_1_d, add_1])

#     conv_3x3_1_e = Conv2D(64, kernel_size = (3, 3), activation = 'relu', padding = 'same')(add_2)
#     bn_4 = BatchNormalization()(conv_3x3_1_e)
#     conv_3x3_1_f = Conv2D(64, kernel_size = (3, 3), activation = 'relu', padding = 'same')(bn_4)
#     add_3 = Add()([conv_3x3_1_f, add_2])
    
    glob_avg_pool = AveragePooling2D((3, 3))(add_1
                                            )
    flatten_ = Flatten()(glob_avg_pool)
    softmax = Dense(10, activation = 'softmax')(flatten_)
    return Model(input_
                 , softmax)

model = resnet_model()
model.summary()


model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(X_train, y_train, epochs = 100, batch_size = 32, validation_split = 0.1)

loss, acc = model.evaluate(X_test, y_test)
print('loss: {}, acc: {}'.format(loss, acc))


#Note1: train with all conv layer (resnet_model) and 100 epochs: acc = .7798 while train-acc: .9941 => overfit
#Note2: train with two first block conv of resnet and 100 epochs: acc = .7295 while train-acc = .9864
#Note3: train with first block conv of restnet (resnet_model_2) and 100 epochs: acc = .7058 while train-acc = .9808
