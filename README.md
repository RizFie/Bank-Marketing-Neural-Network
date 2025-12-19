# Bank Marketing Prediction - Neural Network Classifier
### ISP560: Machine Learning Group Project
 
**Tech Stack:** Python, PyTorch, Scikit-Learn, Pandas  
**Hardware Used:** NVIDIA GeForce RTX 2060 (CUDA 11.8)

---

## 1) Project Overview vs. Weka (Waikato Environment for Knowledge Analysis)
For this assignment, our group chose to implement a **Deep Neural Network (DNN)** using Python and PyTorch, rather than using the standard Weka GUI tools.

**Why we chose this approach:**
1.  **Granular Control:** Weka restricts access to low-level architecture. By coding in PyTorch, we customized layer density, dropout rates (30%), and activation functions manually.
2.  **Data Leakage Prevention:** We implemented advanced **Pipelines** (`sklearn.pipeline`) to ensure scaling and imputation parameters were strictly learned from the Training set and applied to the Test set—something difficult to guarantee in drag-and-drop tools.

---

## 2) The Pipeline

### 1. Data Cleaning
* **Leakage Removal:** Dropped the `duration` column. This variable is only known *after* a call is made, so including it would cause data leakage and fake 100% accuracy.
* **Deduplication:** Removed duplicate customer records.
* **Imputation:**
    * Categorical features (e.g., job, education) filled with **"unknown"**.
    * Numerical features filled with **Mean**.

### 2. Preprocessing
We used a `ColumnTransformer` to automate:
* **StandardScaler:** Normalizing age and economic indicators.
* **One-Hot Encoding:** Converting categorical text into binary vectors.
* **Result:** Input dimensionality expanded from 19 raw features to **62 input neurons**.

### 3. Neural Network Architecture
We built a Feed-Forward Network (`BankModel`) optimized for binary classification:
* **Input Layer:** 60 Neurons
* **Hidden Layer 1:** 30 Neurons + ReLU + **Dropout (0.3)** (Prevents Overfitting)
* **Hidden Layer 2:** 15 Neurons + ReLU + **Dropout (0.3)**
* **Output Layer:** 1 Neuron (Sigmoid Activation)

---

## 3) Results & Performance

We addressed the class imbalance (89% 'No' vs 11% 'Yes') not by oversampling, but by tuning the **Decision Threshold**.

### The "Standard" Approach (Threshold 0.5)
* **Accuracy:** ~90%
* **Recall (Customers Found):** ~21%
* *Critique:* The model was too conservative. It missed 80% of potential subscribers to maintain high accuracy.

### The "Optimized" Approach (Threshold 0.15) - **FINAL MODEL**
We adjusted the sensitivity to catch faint signals from potential customers.
* **Accuracy:** ~86% (Slight trade-off)
* **Recall (Customers Found):** **~63%**
* **Impact:** By sacrificing 4% accuracy, we **tripled** the number of potential leads identified for the bank.

---

## 4) How to Run This Project

### 1. Setup Environment
This project requires a GPU-enabled PyTorch environment (optional but recommended).
```bash
# Install dependencies
pip install -r requirements.txt
```
then run (for CUDA GPU)
```bash
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
```
else
```bash
pip install torch
```

Dataset Information

This project utilizes a modified version of the Bank Marketing Data Set sourced from the UCI Machine Learning Repository.

Original Source: https://archive.ics.uci.edu/dataset/222/bank+marketing

Modifications: The original dataset uses the string 'unknown' to represent missing data in categorical features. For this project, these 'unknown' values have been deleted (replaced with actual null/NaN values) to simulate missing data and facilitate data cleaning and imputation exercises.
