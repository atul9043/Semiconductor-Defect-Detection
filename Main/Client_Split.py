def create_non_iid_clients_fixed(X_train, y_train, num_clients, non_iid_strength, seed, num_classes, img_size):
    """
    FIXED: Create non-IID client datasets with proper data handling
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Group data indices by class
    class_indices = [np.where(y_train == i)[0] for i in range(num_classes)]

    # Generate class distribution for each client using Dirichlet
    class_distribution = np.random.dirichlet([non_iid_strength] * num_classes, num_clients)

    # Assign data indices to each client
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        if len(class_indices[c]) == 0:
            continue

        class_c_indices = class_indices[c].copy()
        np.random.shuffle(class_c_indices)

        # Calculate samples per client for this class
        proportions = class_distribution[:, c]
        num_samples_for_class = (proportions * len(class_c_indices)).astype(int)

        # Ensure all samples are assigned
        remaining = len(class_c_indices) - sum(num_samples_for_class)
        for i in range(remaining):
            num_samples_for_class[i % num_clients] += 1

        # Distribute indices
        start = 0
        for client_id, num_samples in enumerate(num_samples_for_class):
            if num_samples > 0:
                end = start + num_samples
                client_indices[client_id].extend(class_c_indices[start:end])
                start = end

    # Create client datasets
    clients_data = {}
    client_class_distributions = {}

    def client_preprocess(image, label):
        # Convert to float and normalize
        image = tf.cast(image, tf.float32)
        image = image / 2

        # Resize
        image = tf.image.resize(image, [img_size, img_size], method=tf.image.ResizeMethod.BICUBIC)

        # Convert to RGB
        image = tf.repeat(image, repeats=3, axis=-1)

        # Apply contrast
        image = tf.image.adjust_contrast(image, 1.3)
        image = tf.clip_by_value(image, 0.0, 1.0)

        # One-hot encode label
        label_onehot = tf.one_hot(label, depth=num_classes)

        return image, label_onehot

    for client_id in range(num_clients):
        indices = client_indices[client_id]

        if len(indices) > 0:
            # Get client data
            client_X = X_train[indices]
            client_y = y_train[indices]

            # Create dataset
            client_ds = tf.data.Dataset.from_tensor_slices((client_X, client_y))
            client_ds = client_ds.shuffle(1000, seed=seed)
            client_ds = client_ds.map(client_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            client_ds = client_ds.batch(32)
            client_ds = client_ds.prefetch(tf.data.AUTOTUNE)

            clients_data[f"client_{client_id}"] = client_ds

            # Calculate class distribution for analysis
            unique, counts = np.unique(client_y, return_counts=True)
            client_class_dist = dict(zip(unique, counts))
            client_class_distributions[f"client_{client_id}"] = client_class_dist

            print(f"Client {client_id}: {len(indices)} samples, {len(unique)} classes")
        else:
            print(f"⚠️ Client {client_id} has no samples")

    return clients_data, client_class_distributions