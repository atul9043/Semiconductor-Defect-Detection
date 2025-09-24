def main_federated_wafer_pipeline():
    """Main pipeline with better error handling"""
    try:
        # Data preprocessing
        print("Loading and preprocessing wafer defect dataset...")

        clients_data, val_data, test_data, num_classes, train_ids, client_distributions = preprocess_wafer_npz_federated(
            npz_path="/content/drive/MyDrive/Collab_Dataset/Wafer_Map_Datasets.npz",
            img_size=224,
            num_clients=4,
            non_iid_strength=10,
            seed=42
            centralized_debug=True
        )

        print(f"\nDataset prepared for {len(clients_data)} clients with {num_classes} classes")

        # Initialize federated learning
        fed_learner = FederatedWaferLearning(
            num_clients=len(clients_data),
            num_classes=num_classes
        )

        # Run federated learning
        final_model, training_history = fed_learner.run_federated_learning(
            clients_data=clients_data,
            val_data=val_data,
            test_data=test_data,
            num_rounds=10,  # Reduced for testing
            local_epochs=2,
            xai_feedback=False
        )

        # Save model
        try:
            final_model.save('federated_wafer_defect_model.h5')
            print("\n💾 Model saved successfully!")
        except Exception as e:
            print(f"Model saving failed: {e}")

        return final_model, training_history, fed_learner

    except Exception as e:
        print(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

# Execute the pipeline
if __name__ == "__main__":
    model, history, fed_learner = main_federated_wafer_pipeline()