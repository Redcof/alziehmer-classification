import keras.applications


def create_multiview_mobilenet(input_shape, num_views, num_classes, trainable=False):
    """
    Creates a multiview CNN model using MobileNet as the base.

    Args:
        input_shape: Tuple, shape of each input image (e.g., (224, 224, 3)).
        num_views: Integer, number of input views.
        num_classes: Integer, number of output classes.

    Returns:
        A Keras Model instance.
    """
    input_tensor = keras.Input(shape=(num_views, *input_shape), name="multi_view_input")

    # Distribute the input to different branches for each view
    def distribute_views(views):
        unstacked_views = tf.split(views, num_views, axis=1, name="view_input_")
        unstacked_views = [tf.squeeze(t, axis=1) for t in unstacked_views]
        return unstacked_views

    distributed_input = keras.layers.Lambda(distribute_views, name="distribute_view_input")(input_tensor)
    # distributed_input = [keras.Input(shape=input_shape) for _ in range(num_views)]

    # Create a list to store the outputs of each view's MobileNet
    mobilenet_outputs = []

    # Create and apply MobileNet to each view
    for i in range(num_views):
        base_model = keras.applications.MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)
        setattr(base_model, '_name', f"{base_model.name}_view_{i}")
        base_model.trainable = trainable  # Freeze base model weights
        x = base_model(distributed_input[i])
        mobilenet_outputs.append(x)

    # Concatenate the outputs of all views
    merged_features = keras.layers.Concatenate(axis=-1)(mobilenet_outputs)

    # Add custom layers on top of the concatenated features
    x = keras.layers.Conv2D(256, (3, 3), activation='relu')(merged_features)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)

    # Create and compile the final model
    model = keras.Model(inputs=input_tensor, outputs=outputs)
    return model


if __name__ == '__main__':
    import tensorflow as tf

    image_size = 128
    n_ch = 3
    volume_depth = 61
    num_classes = 4
    batch_size = 13
    mvmodel = create_multiview_mobilenet((image_size, image_size, n_ch), volume_depth, num_classes)
    mvmodel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    mvmodel.summary()

    # sample input
    inputs = tf.random.uniform(shape=(batch_size, volume_depth, image_size, image_size, n_ch),
                               minval=-1, maxval=1,
                               dtype=tf.float32)
    # forward pass
    op = mvmodel(inputs)
    assert op.shape == (batch_size, num_classes)
