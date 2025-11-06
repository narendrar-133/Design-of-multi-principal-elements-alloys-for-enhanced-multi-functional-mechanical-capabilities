# 🧪 AI-Driven Material Discovery: Bayesian Optimization for High-Performance Alloys

## Project Overview

This project demonstrates an end-to-end workflow for **AI-driven materials discovery** using machine learning and Bayesian Optimization (BO). The goal is to efficiently screen a large combinatorial space of quaternary (four-element) alloys—specifically, **Nb-based refractory alloys**—to identify new compositions with significantly improved **Specific Strength** ($\text{MP} \cdot \text{cm}^3 \cdot \text{g}^{-1}$).

The workflow leverages the predictive power of a **Gradient Boosting Regressor (GBR)**, trained on experimental data, and couples it with a Bayesian Optimization strategy using the **Expected Improvement (EI)** acquisition function to guide the search for novel, high-performing materials.

---

## 🚀 Key Steps and Methodology

The core of this project is structured into three main phases: **Data Preparation & Modeling**, **Model Evaluation**, and **Bayesian Optimization**.

### 1. Data Preparation & Modeling

* **Data Source:** Experimental data from a dataset of multi-element alloys, focusing on **Specific Strength** as the target property ($\text{Y}$).
* **Feature Engineering:** Raw material compositions were transformed into numerical feature vectors ($\text{X}$) using the **Composition-Based Feature Vectors (CBFV)** library.
    * The **`magpie`** set of elemental properties was used for featurization.
    * The feature set was extended to include **Temperature**, resulting in **133 total features**.
* **Data Preprocessing:**
    * The dataset was split into a **Training Set (80%)** and a **Test Set (20%)** with `random_state=42`.
    * Features were standardized using **`StandardScaler`** (zero mean and unit variance).
* **Model Selection & Optimization:**
    * A **Gradient Boosting Regressor (GBR)** was selected as the base predictive model.
    * **Grid Search Cross-Validation (`GridSearchCV`)** with 3 folds (`cv=3`) was used to find the optimal hyperparameters, maximizing the negative mean squared error (`scoring="neg_mean_squared_error"`).

#### 🏆 Optimized GBR Model

| Hyperparameter | Optimal Value |
| :--- | :--- |
| `learning_rate` | 0.1 |
| `n_estimators` | 50 |
| `min_samples_split` | 10 |
| `max_depth` | 10 |

---

### 2. Model Performance

The optimized GBR model's performance was evaluated using $R^2$ score and Root Mean Squared Error (RMSE).

| Metric | Training Data (Scaled) | Test Data (Scaled) |
| :--- | :--- | :--- |
| **$R^2$ Score** | 0.9785 | 0.8247 |
| **Root Mean Squared Error (RMSE)** | 10.37 $\text{MP} \cdot \text{cm}^3 \cdot \text{g}^{-1}$ | 29.95 $\text{MP} \cdot \text{cm}^3 \cdot \text{g}^{-1}$ |

The $R^2$ score of **0.8247** on the test set confirms the model's strong generalization and suitability for use in the Bayesian Optimization loop.

---

### 3. Bayesian Optimization (BO) for Discovery

Bayesian Optimization aims to efficiently find new high-performing alloys within a vast, un-tested **candidate search space**.

* **Acquisition Function:** **Expected Improvement (EI)** was used to guide the search, balancing predicted performance and uncertainty.
    * The current maximum observed property ($\mu_{\text{max}}$) in the experimental data is **309.7** $\text{MP} \cdot \text{cm}^3 \cdot \text{g}^{-1}$.
* **Candidate Search Space:** A combinatorial search space of **100,000** new alloys was generated. A subset of **1,000** candidates was featurized and scaled for the initial BO step.
* **Prediction & Ranking:** Candidates were ranked by their calculated EI score to identify the most promising alloys for future synthesis.

#### 🔎 Top 5 New Alloy Candidates (Ranked by Expected Improvement)

These candidates represent the highest potential for improved Specific Strength based on the current model's predictions and uncertainty estimates.

| Rank | Formula | Predicted Specific Strength ($\text{MP} \cdot \text{cm}^3 \cdot \text{g}^{-1}$) | Expected Improvement (EI) | Uncertainty (Std) |
| :---: | :--- | :---: | :---: | :---: |
| 1 | $\text{Nb}_{82}\text{Al}_1\text{Co}_1\text{Cr}_{16}$ | 186.32 | $3.40 \times 10^{-5}$ | 28.13 |
| 2 | $\text{Nb}_{81}\text{Al}_1\text{Co}_1\text{Cr}_{17}$ | 187.84 | $3.24 \times 10^{-5}$ | 27.74 |
| 3 | $\text{Nb}_{82}\text{Al}_2\text{Co}_{13}\text{Cr}_3$ | 193.65 | $1.84 \times 10^{-5}$ | 25.82 |
| 4 | $\text{Nb}_{90}\text{Al}_2\text{Co}_7\text{Cr}_1$ | 178.27 | $1.55 \times 10^{-5}$ | 28.87 |
| 5 | $\text{Nb}_{82}\text{Al}_1\text{Co}_2\text{Cr}_{15}$ | 187.88 | $1.41 \times 10^{-5}$ | 26.74 |

---

### 4. Materials Informatics Space Visualization (t-SNE)

A **t-SNE (t-distributed Stochastic Neighbor Embedding)** plot was used to visualize the relationships between the original experimental data (Top/Low Performers) and the new candidates (Top EI) in a reduced-dimensional feature space.



#### 📊 Visualization Interpretation

The visualization of the alloy feature space reveals distinct clustering:
* The **Top Performers** (existing highest-value alloys) and **Low Performers** (existing lowest-value alloys) show significant separation in the feature space.
* The **Top EI** candidates (predicted by BO) form a cluster that is spatially separate from the clusters of both Top and Low Performers.

This separation suggests that the Bayesian Optimization successfully promotes **exploration**, guiding the search towards **new, distinct regions** of the alloy composition space where the model's uncertainty is higher, rather than merely exploiting the immediate neighborhood of known good materials. This is crucial for discovering truly novel, high-performance materials.

---

## 🛠️ Technologies Used

* **Python**
* **scikit-learn:** `GradientBoostingRegressor`, `GridSearchCV`, `StandardScaler`, `PCA`, `TSNE`.
* **pandas & numpy:** Data handling and numerical operations.
* **CBFV:** Composition-Based Feature Vectors for materials featurization.
* **scipy.stats:** Statistical functions for Expected Improvement calculation.
* **matplotlib & seaborn:** Data visualization.

***
