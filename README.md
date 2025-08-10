# ML Projects (UCLA MQE)

A collection of machine learning notebooks completed during the UCLA Master's in Quantitative Economics program. Each notebook focuses on a core ML concept with a small, self-contained exercise.

## Notebooks
- **ML1**: Intro to Colab + basic EDA on PIMA Diabetes (histogram, distribution insights)
- **ML2**: Lasso regression on medical insurance charges (preprocessing, `LassoCV`, in/out predictions)
- **ML3**: Credit card fraud classification with Logistic Regression (ROC curve, threshold for 5% FPR)
- **ML4**: Imbalanced classification comparison (Random Over/Under Sampling, SMOTE) with Logistic Regression
- **ML5**: Decision Tree (max depth 3) on US Permanent Visas dataset
- **ML6**: Neural networks for CLV prediction (GridSearchCV for `MLPRegressor` + simple Keras model)
- **ML7**: RNN (LSTM) predicting next-day stock direction from percentage changes; RMSE comparisons and plots
- **ML8**: Bank marketing classification (preprocessing, SMOTE, decision tree, bagging, confusion matrices)
- **ML9**: K-Means clustering on country-level features (elbow method, 2D visualization, group listings)
- **ML10**: PCA on country data (2 components visualization, loadings heatmap, cumulative explained variance)

## How to run
### Option A — Local (Jupyter)
1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn tensorflow keras yfinance graphviz pydotplus
   ```
3. Launch Jupyter and open any `ML*.ipynb`:
   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```

### Option B — Google Colab
- Open a notebook in Colab (File → Open notebook → GitHub tab or upload).
- Many notebooks expect datasets in Google Drive under `/content/gdrive/MyDrive/ML/`. If your paths differ, either:
  - Mount Drive and place the files in the same folder structure, or
  - Edit the file paths at the top of the notebook cells to where your data lives.

## Datasets referenced
- `diabetes.csv` (PIMA Diabetes)
- `insurance.csv` (Medical insurance charges)
- `fraudTest.csv` (Credit card transactions)
- `us_perm_visas.csv` (zipped as `us_perm_visas.csv.zip`)
- `CLV.csv` (Customer lifetime value)
- `bank-additional-full.csv` (Bank marketing)
- `Country-data.csv` (Country-level indicators)
- Yahoo Finance data via `yfinance` (ticker configurable in ML7)

Note: Paths are often hard-coded for Colab + Google Drive, e.g. `/content/gdrive/MyDrive/ML/...`. Adjust as needed for your environment.

## Notes
- Expect benign warnings from `pandas`, `scikit-learn`, or optimizers (e.g., convergence warnings). They do not materially affect the exercises.
- Some models use randomness; set `random_state` for reproducibility where applicable.

