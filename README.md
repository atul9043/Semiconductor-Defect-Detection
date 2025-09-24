# 🛡️ Federated Learning and XAI for Semiconductor Fault Detection 🔍

A robust, privacy-preserving fault detection system for semiconductor wafer maps using Federated Learning (FL) and Explainable AI (XAI). This project trains a MobileNetV2 model across multiple simulated manufacturing sites ("clients") without centralizing sensitive data and uses Grad-CAM for model transparency.

---

## 📖 Project Overview

This project implements a sophisticated solution for semiconductor fault detection that addresses two key industry challenges: **data privacy** and **model transparency**. By leveraging Federated Learning, the system trains a powerful deep learning model on decentralized data, ensuring sensitive manufacturing information remains secure. Concurrently, Explainable AI (XAI) is integrated to provide clear insights into the model's decision-making process, fostering trust and enabling detailed diagnostics.

---

## ✨ Key Features

* **🌐 Federated Learning**: Trains a global model on decentralized data, ensuring privacy and security.
* **🧠 Explainable AI (XAI)**: Integrates Grad-CAM to visualize and quantify the model's focus, building trust and transparency.
* **📊 Realistic Data Simulation**: Uses a Dirichlet distribution to create realistic, non-IID (Not Independent and Identically Distributed) data distributions across clients.
* **⚖️ Handles Imbalance**: Intelligently manages the severe class imbalance in the dataset using class weighting.
* **🚀 Efficient Architecture**: Employs a custom-headed **MobileNetV2**, a lightweight and efficient CNN ideal for federated environments.
* **⚙️ Automated Training**: Features a robust training pipeline with best practices like Early Stopping and Learning Rate Reduction on Plateau.

---

##  workflow Methodology Workflow

The pipeline follows a robust, end-to-end machine learning process:

1.  **📦 Data Preparation**: Load the raw wafer data, perform domain-specific normalization, and create stratified splits for training, validation, and testing.
2.  **🌍 Federated Distribution**: Distribute the training data across multiple clients using a Dirichlet distribution to simulate a realistic non-IID environment.
3.  **🤝 Federated Training**: Orchestrate the training over multiple rounds using Federated Averaging. In each round, clients train locally, and the server aggregates their updates.
4.  **💡 XAI Feedback**: (Optional) After local training, use the `WaferXAIAnalyzer` to generate and quantify the model's focus, providing insights into the learning process.
5.  **📈 Evaluation**: Evaluate the global model's performance on a centralized validation set after each round and perform a final evaluation on the test set.

---

## 🛠️ Setup and Installation

Follow these steps to set up your local environment.

1.  **Clone the Repository**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Create a Virtual Environment** (Recommended)
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    Your `requirements.txt` file should contain:
    ```
    numpy
    tensorflow
    scikit-learn
    matplotlib
    opencv-python-headless
    tf-keras-vis
    ```

---

## 🚀 How to Run

1.  **Place Your Dataset**: Make sure your `Wafer_Map_Datasets.npz` file is in the project's root directory or update the path in the script.

2.  **Execute the Main Training Pipeline**
    ```bash
    python main_training.py
    ```
    This will run the entire federated learning process and save the best model as `federated_wafer_defect_model_fixed.h5`.

3.  **Execute the Testing Pipeline** (Optional)
    ```bash
    python run_tests.py
    ```
    This will run the systematic validation tests on your data, models, and algorithms.

---

## 🧩 Core Components Explained

* `preprocess_wafer_npz_federated()`: The main data orchestrator. It loads, splits, and preprocesses the data for the federated environment.
* `create_non_iid_clients_fixed()`: The core of the federated simulation, using a Dirichlet distribution to create realistic client datasets.
* `create_federated_mobilenetv2_model()`: Defines the MobileNetV2 architecture with a powerful custom head for the 38-class classification task.
* `WaferXAIAnalyzer()`: The class responsible for all XAI analysis, including generating Grad-CAM heatmaps and quantifying the model's focus.
* `FederatedWaferLearning()`: The "brain" of the simulation, managing the training rounds, client updates, and global model aggregation.

---

## 🎯 Performance Benchmark

The project's performance can be benchmarked against the base paper. The paper achieved a **test accuracy of 96.81%** with MobileNetV2 over 10 federated rounds. This serves as a strong benchmark for this implementation.

---

## ©️ Citation

This project builds upon the foundational concepts and benchmarks established in the following paper:

> Patel, T., Murugan, R., Yenduri, G., Jhaveri, R. H., Snoussi, H., & Gaber, T. (2024). Demystifying Defects: Federated Learning and Explainable AI for Semiconductor Fault Detection. *IEEE Access*, 12, 116987-117007. DOI: 10.1109/ACCESS.2024.3425226.