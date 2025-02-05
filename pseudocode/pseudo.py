def initialize():
    ratio = (0.7, 0.2, 0.1)
    batch_size = 13
    train_data, validation_data, test_data = get_oasis_dataset(batch_size, ratio)

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_data = RandomVerticalFlip(train_data)
    train_data = RandomHorizontalFlip(train_data)
    train_data = bilateral_filter(train_data)
    train_data = Standardize(train_data, mean=imagenet_mean, std=imagenet_std)

    validation_data = Standardize(validation_data, mean=imagenet_mean, std=imagenet_std)
    test_data = Standardize(test_data, mean=imagenet_mean, std=imagenet_std)


def create_multiview_model():
    volume_depth = 61
    num_classes = 4
    multiview_output = []
    for i in range(volume_depth):
        backbone = pretrained_mobilenet()
        multiview_output.append(backbone)
    merged_features = Concatenate(multiview_output)
    feature = Conv2D(merged_features, 256, (3, 3), activation='relu')
    feature = MaxPooling2D(feature, pool_size=(2, 2))
    feature = GlobalAveragePooling2D(feature)
    feature = Dropout(feature, probability=0.5)
    feature = Dense(feature, num_classes, activation='softmax')
    return feature


def experiment():
    multiview_mobilenet = create_multiview_model()

    learning_rate = 0.001
    loss_function = "categorical_cross_entropy"
    optimization_algorithm = "adam"

    multiview_mobilenet_model = compile(multiview_mobilenet, learning_rate, loss_function)

    train_metrics = train(multiview_mobilenet_model, train_data, optimization_algorithm)
    validation_metrics = multiview_mobilenet_model.forward(validation_data)
    test_metrics = multiview_mobilenet_model.forward(test_data)

    save_model(multiview_mobilenet_model)
    save_metrics(train_metrics, validation_metrics, test_metrics)
