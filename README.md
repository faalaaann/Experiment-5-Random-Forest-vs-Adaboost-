# Experiment 5: Random Forest vs AdaBoost

This experiment compares **Random Forest**, **AdaBoost**, and a small **TensorFlow/Keras neural network** on the Breast Cancer Wisconsin diagnostic dataset. The notebook covers data exploration, preprocessing, model training, cross-validation, evaluation, confusion matrices, and Random Forest feature importance.

## Objectives

- Implement and compare ensemble learning methods.
- Evaluate models using accuracy, precision, recall, and F1-score.
- Use cross-validation to select Random Forest hyperparameters and assess AdaBoost.
- Visualize confusion matrices, model performance, training history, and feature importance.

## Dataset

The dataset is loaded through `sklearn.datasets.load_breast_cancer` and contains:

- **569 samples**
- **30 numeric features** describing cell nuclei
- **2 classes:** malignant and benign
- **No missing values**

The data is split into 80% training and 20% testing sets using stratification. Features used by the neural network are standardized with `StandardScaler`.

## Models

### Random Forest

A Random Forest classifier is evaluated with 5-fold cross-validation across several parameter choices:

- `n_estimators`: 100, 200, or 300
- `max_depth`: 4, 6, or unlimited

The experiment selects the best configuration based on mean cross-validation accuracy.

### AdaBoost

AdaBoost uses decision stumps (`max_depth=1`) as weak learners with 200 estimators and a learning rate of 1.0.

### Neural Network

The neural-network baseline is a Keras sequential model with two hidden layers:

- 32 ReLU units
- 16 ReLU units
- 1 sigmoid output unit

It is trained with Adam and binary cross-entropy for 50 epochs.

## Results

Results recorded from the notebook's saved run:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Random Forest | 0.9474 | 0.9583 | 0.9583 | 0.9583 |
| AdaBoost | 0.9649 | 0.9474 | 1.0000 | 0.9730 |
| Neural Network | 0.9737 | 0.9859 | 0.9722 | 0.9790 |

The neural network achieved the highest test accuracy in this run. AdaBoost achieved perfect recall for the positive class, while Random Forest provided interpretable feature-importance values.

The top Random Forest features included **worst perimeter**, **worst area**, **worst concave points**, and **mean concave points**.

## Requirements

- Python 3.9+
- NumPy
- pandas
- Matplotlib
- Seaborn
- scikit-learn
- TensorFlow
- Jupyter Notebook or JupyterLab

Install the dependencies with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow jupyter
```

## Running the experiment

1. Clone the repository:

   ```bash
   git clone https://github.com/faalaaann/Experiment-5-Random-Forest-vs-Adaboost-.git
   cd Experiment-5-Random-Forest-vs-Adaboost-
   ```

2. Start Jupyter:

   ```bash
   jupyter notebook
   ```

3. Open **`EX5 Random forest vs adaboost.ipynb`** and run the cells from top to bottom.

The notebook uses random seed `42` to make the scikit-learn and TensorFlow results reasonably reproducible. Neural-network results may vary slightly across hardware and TensorFlow versions.

## Repository contents

- `EX5 Random forest vs adaboost.ipynb` — complete executable experiment
- `README.md` — project documentation

## License

This project is intended for educational and experimental use.
