from keras import Sequential, layers
from keras.src.initializers.initializers import Constant


def cnn_3d_classifier(input_shape=(16, 16, 16, 1), num_classes=10, activation="softmax"):
    model = Sequential()
    model.add(
        layers.Conv3D(32, (3, 3, 3), activation='relu', input_shape=input_shape, bias_initializer=Constant(0.01)))
    model.add(layers.Conv3D(32, (3, 3, 3), activation='relu', bias_initializer=Constant(0.01)))
    model.add(layers.MaxPooling3D((2, 2, 2)))
    model.add(layers.Conv3D(64, (3, 3, 3), activation='relu'))
    model.add(layers.Conv3D(64, (2, 2, 2), activation='relu'))
    model.add(layers.MaxPooling3D((2, 2, 2)))
    model.add(layers.Dropout(0.6))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, 'relu'))
    model.add(layers.Dropout(0.7))
    model.add(layers.Dense(128, 'relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation))
    model.summary()
    return model
