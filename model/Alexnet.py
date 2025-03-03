from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D,Dense,Flatten,Dropout

def Alexnet(inputshape = (224,224,3), outputshape=2):

    model = Sequential()
    model.add(Conv2D(
        filters=96,
        kernel_size=(11,11),
        strides=(4,4),
        padding='same',
        activation='relu',
        input_shape=inputshape
    ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(
        pool_size=(3,3),
        strides=(2,2),
        padding='valid'
    ))

    model.add(
        Conv2D(
            filters=256,
            kernel_size=(5,5),
            padding='same',
            activation='relu'
        )
    )
    model.add(BatchNormalization())
    model.add(
        MaxPooling2D(
            pool_size=(3,3),
            strides=(2,2),
            padding='valid'
        )
    )

    model.add(
        Conv2D(
            filters=384,
            kernel_size=(3,3),
            padding='same'
        )
    )
    model.add(BatchNormalization())
    model.add(
        Conv2D(
            filters=384,
            kernel_size=(3,3),
            padding='same'
        )
    )
    model.add(BatchNormalization())

    model.add(
        Conv2D(
            filters=256,
            kernel_size=(3,3),
            padding='same'
        )
    )
    model.add(BatchNormalization())
    model.add(
        MaxPooling2D(
            pool_size=(3,3),
            strides=(2,2),
        )
    )
    model.add(Flatten())
    model.add(Dense(9216,activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(4096,activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(2048,activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(1024,activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(outputshape, activation='softmax'))


    return model

model = Alexnet()
model.summary()