##  Customer Churn Classification Project

This is my **final project** from a machine learning course, where I applied everything I learned by building a real-world churn prediction system.

---

###  Project Overview

This project predicts whether a bank customer is likely to **churn** (leave the bank) based on personal and account data. I approached the problem using both **classical ML models** and a **neural network (ANN)**, while optimizing performance and usability.

---

###  Project Workflow

####  1. **Data Preprocessing**

* Cleaned and prepared a customer dataset
* Handled feature scaling, encoding, and class imbalance
* Applied proper transformations to match model inputs (especially for Streamlit)

####  2. **Model Selection with PyCaret**

* Used `pycaret.classification` to:

  * Automate model comparison
  * Rank classifiers (Random Forest, SVM, XGBoost, ANN, e.g.)
  * Identify top 3 models based on accuracy and AUC
  * Exported hyperparameters from PyCaret for custom model building

#### 3. **Custom Modeling**

* Trained **Random Forest** using parameters optimized by PyCaret
* Trained a **Deep Learning ANN** using TensorFlow/Keras
* Tracked metrics including accuracy, F1-score, precision, recall, and AUC
* Prevented overfitting using:

  * Class weights
  * Early stopping
  * Validation curves
  * ROC analysis

#### 4. **Web App Interface**

* Built a simple and interactive **Streamlit app** for customer input
* Used the trained Random Forest model to predict churn
* Applied live preprocessing (one-hot encoding, scaling) before prediction
* Added UX feedback (probabilities, result summary)

---

### Platforms Used :

* **Google Colab**: For training heavy models (PyCaret, ANN)
* **Local Machine**: For model testing, tuning, and Streamlit frontend
* **Conda & Virtualenv** for environment isolation
*   ### Requirements

---

  I’ve included:

   * `requirements.txt`: For all packages used on local hardware
   * `.pkl` or `.h5` model files: For deployment
   * Trained `scaler.pkl`: To match preprocessing with training

---

###  How to Run the App

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

### Files Overview

| File/Folder                | Description                             |
| -------------------------- | --------------------------------------- |
| `churn_preprocessed.csv`   | Cleaned dataset                         |
| `model.pkl / ann_model.h5` | Trained model (RF or ANN)               |
| `scaler.pkl`               | Fitted scaler for input standardization |
| `streamlit_app.py`         | Web app UI using Streamlit              |
| `requirements.txt`         | Local environment dependencies          |

---

**Hossein**
Final project for ML course | Practicing real-world deployment

