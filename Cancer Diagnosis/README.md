# Deep Learning PR 1 — Breast Cancer Classification

Red & White Skill Education · Deep Learning

A step-by-step deep learning project that classifies tumours as **Malignant (0)** or **Benign (1)**.
It starts with a Single-Layer Perceptron and adds one technique at a time to see what each one changes.

## Dataset
Breast Cancer Wisconsin (Diagnostic) — loaded directly from sklearn with `load_breast_cancer(as_frame=True)`.
569 samples (212 Malignant, 357 Benign), 30 numeric features, no missing values. No CSV download is needed.
(Also available at the [UCI ML Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).)

## Topics covered
1. **Data scaling** — `StandardScaler` fitted on the training set only, so gradient-based optimisers converge faster and avoid data leakage.
2. **SLP** — one sigmoid neuron (31 parameters) used as a linear baseline.
3. **MLP** — hidden layers (64 → 32) that learn non-linear patterns; parameter count checked by hand.
4. **Activation functions** — ReLU vs Tanh vs Sigmoid in the hidden layers.
5. **Early Stopping** — `monitor`, `patience`, `restore_best_weights` explained and compared with training without it.
6. **Dropout** — rates 0.1, 0.3 and 0.5 compared.
7. **Regularization** — L1, L2 and L1-L2 (ElasticNet), plus a final model combining Dropout + L2 + Early Stopping.

## Plots
| Plot | Caption |
| --- | --- |
| ![](plots/class_distribution.png) | Class distribution: 212 Malignant vs 357 Benign |
| ![](plots/correlation_heatmap.png) | 30×30 feature correlation heatmap |
| ![](plots/slp_training_curves.png) | SLP training vs validation loss and accuracy |
| ![](plots/slp_confusion_matrix.png) | SLP confusion matrix on the test set |
| ![](plots/activation_comparison.png) | MLP accuracy with ReLU vs Tanh vs Sigmoid hidden layers |
| ![](plots/mlp_confusion_matrix.png) | Best MLP (ReLU) confusion matrix on the test set |
| ![](plots/early_stopping_curves.png) | Early Stopping: validation loss with the stopping epoch marked |
| ![](plots/early_stopping_comparison.png) | Validation loss with vs without Early Stopping |
| ![](plots/dropout_rate_comparison.png) | Validation accuracy for dropout rates 0.1 / 0.3 / 0.5 |
| ![](plots/regularization_comparison.png) | Training vs validation loss for L1 / L2 / L1-L2 |
| ![](plots/regularization_accuracy_bar.png) | Test accuracy: no regularization vs L1 vs L2 vs L1-L2 |

## Results
Precision, recall and F1 are calculated for the **Malignant** class (missing a malignant case is the costly mistake).

| Model | Architecture | Regularization | Dropout Rate | Early Stopping | Test Accuracy | Test Precision | Test Recall | Test F1-Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SLP | 30-1 | None | 0.0 | No | 0.9474 | 0.9500 | 0.9048 | 0.9268 |
| MLP-ReLU | 30-64-32-1 | None | 0.0 | No | 0.9649 | 0.9318 | 0.9762 | 0.9535 |
| MLP + Early Stopping | 30-128-64-1 | None | 0.0 | Yes (patience=15) | 0.9649 | 0.9318 | 0.9762 | 0.9535 |
| MLP + Dropout (0.5) | 30-128-64-1 | None | 0.5 | Yes (patience=20) | 0.9561 | 0.9111 | 0.9762 | 0.9425 |
| MLP + L2 | 30-128-64-1 | L2 (0.001) | 0.0 | Yes (patience=20) | 0.9561 | 0.9111 | 0.9762 | 0.9425 |
| Final Combined Model | 30-128-64-1 | L2 (0.001) | 0.3 | Yes (patience=20) | 0.9561 | 0.9111 | 0.9762 | 0.9425 |

The test set has 114 samples, so one sample changes the accuracy by about 0.9%. The biggest improvement is from SLP to MLP (fewer missed malignant cases).

## Tools used
Python, TensorFlow / Keras, scikit-learn, pandas, numpy, matplotlib, seaborn

