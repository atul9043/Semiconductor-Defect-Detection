def create_federated_mobilenetv2_model(input_shape=(224, 224, 3), num_classes=38):
    """
    FIXED: MobileNetV2 model with proper layer handling
    """
    # Data augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomFlip("vertical"),
        tf.keras.layers.RandomContrast(0.2),
        tf.keras.layers.RandomBrightness(0.15),
    ], name="federated_wafer_augmentation")

    inputs = tf.keras.Input(shape=input_shape, name="input_layer")
    x = data_augmentation(inputs)

    # Load MobileNetV2 base
    mobilenet_base = MobileNetV2(
        input_shape=input_shape,
        weights=None,  # No pretrained weights for better federated learning
        include_top=False,
        alpha=1.0
    )
    mobilenet_base.trainable = True

    # FIXED: Properly apply MobileNetV2 without hardcoded training parameter
    x = mobilenet_base(x)

    # Add custom head
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.BatchNormalization(name="bn_1")(x)
    x = tf.keras.layers.Dropout(0.3, name="dropout_1")(x)

    x = tf.keras.layers.Dense(512, activation='relu', name="dense_1")(x)
    x = tf.keras.layers.BatchNormalization(name="bn_2")(x)
    x = tf.keras.layers.Dropout(0.5, name="dropout_2")(x)

    x = tf.keras.layers.Dense(256, activation='relu', name="dense_2")(x)
    x = tf.keras.layers.Dropout(0.3, name="dropout_3")(x)

    # Final classification layer
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name="predictions")(x)

    model = tf.keras.Model(inputs, outputs, name="FederatedWaferMobileNetV2")

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')
        ]
    )

    return model
