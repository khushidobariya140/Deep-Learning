# Deep Learning PR 2 — Adult Income Classification

**Institute:** Red & White Skill Education  
**Subject:** Deep Learning  
**Task:** Binary classification — predict whether annual income exceeds $50K.

## Project Overview
This project uses a Multilayer Perceptron (MLP) with TensorFlow/Keras to classify records from the Adult Census Income dataset. It studies activation functions, weight initialization, loss functions, Batch Normalization and optimizers.

## Objectives
- Clean and explore the Adult Income dataset.
- Build a reusable `build_ann()` function and baseline ANN.
- Compare ReLU, Tanh, Sigmoid and ELU.
- Compare Glorot Uniform/Normal, He Uniform/Normal and zero initialization.
- Compare BCE, MSE, Weighted BCE and Focal Loss.
- Study Batch Normalization placement and learned gamma/beta.
- Compare SGD, momentum, RMSprop and Adam, including learning-rate sensitivity.
- Evaluate using accuracy, class-1 precision/recall/F1 and ROC-AUC.

## Dataset
**Adult Income (Census Income)** — UCI Machine Learning Repository. The dataset is based on 1994 US Census records.
- UCI: https://archive.ics.uci.edu/dataset/2/adult
- Kaggle CSV: https://www.kaggle.com/datasets/wenruliu/adult-income-dataset
- Expected input filename: `adult.csv` (merged train + test CSV, 48,842 rows × 15 columns).
- Target: `income` (`<=50K` → 0, `>50K` → 1).

**Dataset is not bundled** because it must be downloaded from the source. Place `adult.csv` in this project folder before running the notebook.

## Preprocessing Workflow
1. Strip whitespace from string columns.
2. Replace `?` with `NaN`; inspect missing values and drop rows containing missing data, as required by the assignment.
3. Drop `fnlwgt` and redundant `education`; retain `education.num`.
4. Encode `income` as 0/1.
5. Split into train, validation and test partitions using stratification.
6. Apply `pd.get_dummies(drop_first=True)` and align validation/test columns to the training schema.
7. Fit `StandardScaler` on training numeric columns only, then transform validation/test. One-hot binary columns are not scaled.

> **Leakage prevention:** The assignment text describes encoding/scaling before splitting. This implementation performs the split first and fits preprocessing only on training data. It preserves the requested transformations while avoiding information leakage.

## Technologies
- Python 3.10+
- TensorFlow 2.x / Keras
- scikit-learn
- pandas, NumPy
- Matplotlib, Seaborn
- Jupyter Notebook

## Project Structure
```text
DL_PR2_Project/
├── DL_PR2.ipynb
├── README.md
├── requirements.txt
├── adult.csv                 # download separately; not included
├── model_results_all.csv    # generated when notebook runs
├── model_results_summary.csv# generated when notebook runs
└── plots/                    # generated figures are saved here
```

## Installation
Create and activate a virtual environment, then run:
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
pip install -r requirements.txt
```

## Execution
1. Download the merged Adult Income CSV and save it as `adult.csv` beside the notebook.
2. Start Jupyter: `jupyter notebook` (or open the notebook in VS Code).
3. Open `DL_PR2.ipynb`.
4. Choose **Kernel → Restart & Run All**.
5. Wait for all experiments to finish. This notebook trains many models and may take time, especially on CPU.
6. Inspect the generated `plots/` figures and `model_results_*.csv` files.
7. Export HTML: `jupyter nbconvert --to html DL_PR2.ipynb --output DL_PR2.html`.

## Workflow / Experiments
| Task | Experiments |
|---|---|
| 1 | Loading, cleaning, target encoding, class balance, four EDA plots, leakage-safe OHE/scaling and stratified split |
| 2 | Baseline 128→64→1 ANN, parameter verification, training curves, classification report and confusion matrix |
| 3 | Four activations, ReLU activation distribution/dead fraction, sigmoid gradient diagnostic |
| 4 | Five initializers, convergence speed, zero-initializer failure and weight histograms |
| 5 | BCE vs MSE, Weighted BCE, custom Focal Loss and minority-class metric comparison |
| 6 | BatchNorm vs baseline, BN placement and gamma/beta inspection |
| 7 | Five optimizer variants, learning-rate sensitivity, final combined model, result table and ROC curves |

## Visualizations
All plots are saved in `/plots` when the notebook is executed. Key figures include:
- `eda_class_balance.png`, `eda_age_by_income.png`, `eda_hours_by_income.png`, `eda_education_by_income.png`, `eda_correlation_heatmap.png`
- `activation_comparison.png`, `relu_activation_distribution.png`, `sigmoid_gradient_magnitudes.png`
- `initialiser_convergence.png`, `zeros_initialization_failure.png`, `weight_distributions.png`
- `loss_f1_comparison.png`, `weighted_bce_metrics.png`
- `batchnorm_dynamics.png`, `batchnorm_position.png`, `batchnorm_gamma_beta.png`
- `optimiser_convergence.png`, `learning_rate_sensitivity.png`, `results_table.png`, `roc_curves.png`

## Results
**Actual metrics are intentionally not stated in this README before running the notebook on the supplied dataset.** After execution, use `model_results_summary.csv` and the notebook's highlighted results table to report the measured accuracy, class-1 precision, recall, F1 and ROC-AUC. Do not copy expected values from the assignment brief as if they were measured results.

## Conclusion
The notebook provides a controlled comparison of common ANN design choices for imbalanced tabular binary classification. The best configuration must be selected from the observed validation metrics and reported once on the held-out test set. The Adult dataset is historical and contains sensitive demographic attributes; this educational model must not be used for real employment, credit or eligibility decisions.

## Video
**Video link:** Add your unlisted YouTube or Google Drive link here after recording the required 5–10 minute face + screen explanation.

## GitHub Submission Notes
Include `DL_PR2.ipynb`, exported `DL_PR2.html`, `README.md`, `requirements.txt`, and the generated `plots/` directory. Use meaningful commits for data cleaning, activation/initializer experiments, loss/BatchNorm experiments and optimizer/final evaluation.
