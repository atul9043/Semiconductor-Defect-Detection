def preprocess_wafer_npz_federated(
        npz_path,
        img_size=224,
        num_clients=4,
        test_size=0.15,
        val_size=0.15,
        seed=42,
        non_iid_strength=10,
        centralized_debug=True
    ):
    """
    FIXED: Robust preprocessing pipeline that avoids None values in tf.data
    """
    # Step 1: Load and validate data
    data = np.load(npz_path)
    images, labels_onehot = data["arr_0"], data["arr_1"]

    print(f"Loaded data: {images.shape}, Labels: {labels_onehot.shape}")

    # Ensure images have the right shape
    if images.ndim == 3:
        images = np.expand_dims(images, -1)


    # Convert one-hot labels to class indices
    label_tuples = [tuple(l) for l in labels_onehot]
    unique_labels, label_ids = np.unique(label_tuples, axis=0, return_inverse=True)
    num_classes = len(unique_labels)

    print(f"Found {num_classes} classes")

    # Step 2: Split data using sklearn first (more reliable)
    X_temp, X_test, y_temp, y_test = train_test_split(
        images, label_ids, test_size=test_size, random_state=seed, stratify=label_ids
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size/(1-test_size), random_state=seed, stratify=y_temp
    )

    print(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # Step 3: Create preprocessing function
    def preprocess_image(image, label):
        # Convert to float and normalize
        image = tf.cast(image, tf.float32)
        image = image / 2  # Simple normalization instead of /2.0

        # Resize
        image = tf.image.resize(image, [img_size, img_size], method=tf.image.ResizeMethod.BICUBIC)

        # Convert to RGB
        image = tf.repeat(image, repeats=3, axis=-1)

        # Apply contrast adjustment
        image = tf.image.adjust_contrast(image, 1.3)
        image = tf.clip_by_value(image, 0.0, 1.0)

        # Convert label to one-hot
        label_onehot = tf.one_hot(label, depth=num_classes)

        return image, label_onehot

    # Step 4: Create validation and test datasets
    def create_dataset(X, y, batch_size=32, shuffle=False):
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        if shuffle:
            dataset = dataset.shuffle(1000, seed=seed)
        dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    val_ds = create_dataset(X_val, y_val, batch_size=32, shuffle=False)
    test_ds = create_dataset(X_test, y_test, batch_size=32, shuffle=False)

    if centralized_debug:
        print("\n⚡ Running centralized debug training...")
        train_ds = create_dataset(X_train, y_train, shuffle=True)

        model = create_federated_mobilenetv2_model(
            input_shape=(img_size, img_size, 3),
            num_classes=num_classes
        )

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=5,
            verbose=1
        )

        print("Centralized debug training finished.")
        print(f"Final Val Accuracy: {history.history['val_accuracy'][-1]:.4f}")

    # Step 5: Create federated clients
    clients_data, client_class_distributions = create_non_iid_clients_fixed(
        X_train, y_train, num_clients, non_iid_strength, seed, num_classes, img_size
    )



    return clients_data, val_ds, test_ds, num_classes, y_train, client_class_distributions