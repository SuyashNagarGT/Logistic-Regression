# 🔠 Classification — Teaching Models to Decide  
> **“When the question isn’t how much, but which one.”** 😅📊

Classification is a **supervised learning task** where the goal is to assign inputs to **discrete categories**.  
Instead of predicting a number (like regression), classification predicts a **label** — yes/no, spam/not spam, cat/dog, etc.

---

<details>
<summary>🧠 What Is Classification?</summary>

### 🎯 Goal  
Predict the **class label** for a given input based on learned patterns.

### 🧪 Types  
- **Binary Classification**: Two classes (e.g., spam vs not spam)  
- **Multiclass Classification**: More than two classes (e.g., mood: happy, sad, neutral)  
- **Multilabel Classification**: Multiple labels per instance (e.g., tags on a blog post)

### ☕ Analogy  
Imagine Kanak predicting whether a GitHub commit is “stable” or “risky.” That’s binary classification.  
If Kanak predicts the commit type — “bug fix,” “feature,” or “refactor” — that’s multiclass.

</details>

---

<details>
<summary>📊 Common Classification Algorithms</summary>

| Algorithm              | Description                                 |
|------------------------|---------------------------------------------|
| **Logistic Regression** | Linear decision boundary for binary tasks  |
| **Decision Trees**      | Rule-based splits for interpretability      |
| **Random Forest**       | Ensemble of trees for robustness            |
| **Gradient Boosting**   | Sequential learners correcting errors       |
| **Support Vector Machine (SVM)** | Maximizes margin between classes |
| **K-Nearest Neighbors (KNN)** | Classifies based on closest examples |
| **Naive Bayes**         | Probabilistic model based on Bayes’ theorem |
| **Neural Networks**     | Deep learning for complex patterns          |

</details>

---

<details>
<summary>🧪 Example: Predicting Mood from Stress Level</summary>

```python
from sklearn.linear_model import LogisticRegression

X = [[1], [2], [3], [4], [5]]  # Stress level
y = [0, 0, 1, 1, 1]            # Mood: 0 = Calm, 1 = Stressed

model = LogisticRegression()
model.fit(X, y)

print("Predicted mood:", model.predict([[3]])[0])
print("Probability of stress:", model.predict_proba([[3]])[0][1])
```
</details>

---

<details> <summary>🏭 Industry Use Cases</summary>

🏥 Healthcare
Disease diagnosis (e.g., cancer detection)

Patient risk stratification

Medical image classification

🏦 Finance
Credit approval (good vs bad risk)

Fraud detection

Loan default prediction

📧 Email & Security
Spam filtering

Malware classification

Intrusion detection

🛍️ Retail & E-commerce
Customer churn prediction

Product categorization

Sentiment analysis on reviews

🎓 Education
Student performance classification

Dropout risk prediction

Exam pass/fail prediction

📈 Marketing
Lead scoring

Campaign response prediction

Customer segmentation

</details>

---
<details>
  
<summary>📏 Evaluation Metrics</summary>

### 📊 Classification Evaluation Metrics

| Metric      | Description                                      |
|-------------|--------------------------------------------------|
| Accuracy    | % of correct predictions                         |
| Precision   | % of predicted positives that are correct        |
| Recall      | % of actual positives that were found            |
| F1 Score    | Harmonic mean of precision and recall            |
| ROC-AUC     | Ability to rank predictions correctly            |

Use `confusion_matrix`, `classification_report`, and `roc_curve` from `sklearn.metrics` for diagnostics.

</details>
