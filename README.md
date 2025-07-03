# 📱 SMS Spam Detection using Machine Learning

This project builds a machine learning model to classify SMS messages as **spam** or **ham** (not spam) using the [UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection). It uses natural language processing (TF-IDF) and multiple classification models to identify the best performing algorithm.

---

## 📦 Dataset

- **Name:** SMS Spam Collection
- **Source:** UCI Machine Learning Repository
- **Format:** Tab-separated file with 2 columns:
  - `label`: `spam` or `ham`
  - `message`: The SMS message text

---

## 📊 Workflow

1. **Download and load the dataset**
2. **Text preprocessing** and label encoding
3. **Vectorization** using TF-IDF
4. **Train models:**
   - Naive Bayes
   - Logistic Regression
   - Random Forest
   - Support Vector Machine (SVM)
5. **Evaluate models** using precision, recall, and F1-score
6. **Save the best model** (SVM) and the TF-IDF vectorizer

---

## 🧪 Model Results

| Model               | Precision | Recall | F1-Score |
|--------------------|-----------|--------|----------|
| **SVM**             | **0.984** | **0.984** | **0.983** |
| Random Forest       | 0.976     | 0.975     | 0.974     |
| Naive Bayes         | 0.971     | 0.970     | 0.969     |
| Logistic Regression | 0.969     | 0.968     | 0.966     |

✅ **Best Model:** SVM (Linear Support Vector Machine)

---

## 🧠 Requirements

This project uses the following Python libraries:

- `pandas`
- `scikit-learn`
- `joblib`
- `wget` (for downloading data in Colab)

---

## 🚀 How to Run in Google Colab

1. Open the notebook in Google Colab
2. Run all cells in order
3. The script will:
   - Download the dataset
   - Train and evaluate models
   - Save the best model as `svm_spam_model.joblib`
   - Save the vectorizer as `tfidf_vectorizer.joblib`

---

## 💡 Predicting New SMS

You can load the saved model and vectorizer to classify new messages:

```python
import joblib

model = joblib.load("svm_spam_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

msg = ["You've won a free iPhone! Click here to claim."]
msg_vec = vectorizer.transform(msg)
prediction = model.predict(msg_vec)

print("Spam" if prediction[0] == 1 else "Ham")
