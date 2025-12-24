# PartialGC: Proactive Block-level Page Migration for Superblock Management in Modern SSDs
## 1. Requirements
python 3.8
scikit-learn 1.1.1
pandas 1.4.2

## 2. Model
tree_model.py: The relevant code and parameters for the tree model. 

Incremental_model.py: The relevant code and parameters for the incremental learning 

## 3. Data
train_data.xlsx: The dataset for model training. 

predict_data.xlsx: The dataset for model predicting.

## 4. Script

In SSDSim, we use **Cython** to call Python functions for model prediction and incremental learning.

trace_process.c: This function is used to collect data during trace runtime.

model_simulation.c: This code uses Cython to import a Python file as a external library during SSDSim runtime, enabling it to call the model for prediction.
