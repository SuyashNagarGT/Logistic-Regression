# 🔠 Logistic Regression vs Linear Regression  
> **“When the goal shifts from predicting a number to making a decision.”** 😅📊

---

<details>
<summary>🧠 What Is Logistic Regression?</summary>

### 🎯 Purpose  
Logistic Regression is a **classification algorithm** used to predict the **probability** of a categorical outcome — typically binary (e.g., yes/no, spam/not spam).
It’s linear in the inputs, but nonlinear in the output — thanks to the **sigmoid function**.

### 📐 Core Mechanism  
Instead of fitting a straight line, it fits a **sigmoid curve** that maps any input to a value between 0 and 1:



\[
\sigma(z) = \frac{1}{1 + e^{-z}}, \quad \text{where } z = \beta_0 + \beta_1 x_1 + \dots + \beta_n x_n
\]



This output is interpreted as a **probability**, and a threshold (usually 0.5) is used to assign a class label.

### ☕ Analogy  
Imagine Kanak predicting whether a GitHub commit is “stable” or “risky.” Logistic regression doesn’t give a score — it gives a **confidence level**.

</details>

---

<details>
<summary>📈 Why Not Use Linear Regression for Classification?</summary>

### ❌ Linear Regression Limitations  
- Predicts **continuous values**, not probabilities  
- Outputs can be **less than 0 or greater than 1**, which makes no sense for class probabilities  
- Doesn’t model the **decision boundary** between classes  
- Sensitive to **outliers** and **imbalanced data**

### 🧠 Logistic Regression Advantages  
- Outputs are **bounded between 0 and 1**  
- Models **probabilities**, not raw scores  
- Can be extended to **multiclass** and **multilabel** problems  
- Supports **regularization** (L1, L2) for feature selection and generalization

</details>

---

<details>
<summary>🧪 Example Comparison</summary>

```python
from sklearn.linear_model import LinearRegression, LogisticRegression

X = [[1], [2], [3], [4], [5]]
y_class = [0, 0, 1, 1, 1]  # Classification target
y_reg = [2, 3, 4, 6, 7]    # Regression target

# Linear Regression
lin_model = LinearRegression()
lin_model.fit(X, y_reg)
print("Linear prediction:", lin_model.predict([[3]])[0])

# Logistic Regression
log_model = LogisticRegression()
log_model.fit(X, y_class)
print("Logistic prediction:", log_model.predict([[3]])[0])
print("Probability of class 1:", log_model.predict_proba([[3]])[0][1])

```
</details>

---

<details>
<summary>🔠 Types of Logistic Regression — Use Case Guide </summary>


| Type                     | Description                             | Common Use Cases                          | When to Use                                      | When to Avoid                                      |
|--------------------------|-----------------------------------------|-------------------------------------------|--------------------------------------------------|----------------------------------------------------|
| **Binary Logistic**      | Predicts between two classes (0 or 1)   | Spam detection, disease diagnosis, churn  | When the target has exactly two categories       | When target has more than two classes              |
| **Multiclass Logistic**  | Predicts one out of multiple classes    | Mood classification, product category     | When classes are mutually exclusive and >2       | When classes are ordered or overlapping            |
| **Ordinal Logistic**     | Predicts ordered categories             | Satisfaction levels, credit ratings       | When target classes have a natural order         | When order doesn’t matter or classes are nominal   |
| **Multilabel Logistic**  | Predicts multiple labels per instance   | Text tagging, image classification        | When each input can belong to multiple categories| When labels are mutually exclusive                 |

</details>

---

<details>
<summary>📊 Evaluation Metrics</summary>

| Metric       | What It Measures                                | Best Used For                      | Avoid When...                             | Good vs Bad Example                      |
|--------------|--------------------------------------------------|------------------------------------|-------------------------------------------|------------------------------------------|
| **Accuracy** | % of correct predictions                         | Balanced classification problems   | Classes are imbalanced                    | ✅ 95% = strong; ❌ 60% = weak             |
| **Precision**| % of predicted positives that are correct        | False positives are costly         | You care more about catching all positives| ✅ 0.90 = few false alarms; ❌ 0.40 = noisy|
| **Recall**   | % of actual positives that were found            | False negatives are costly         | You want fewer false alarms               | ✅ 0.85 = good coverage; ❌ 0.30 = misses  |
| **F1 Score** | Harmonic mean of precision and recall            | Imbalanced classification          | Precision and recall are both very high   | ✅ 0.88 = balanced; ❌ 0.50 = inconsistent |
| **ROC-AUC**  | Ability to rank predictions correctly            | Binary classification with probabilities | You need hard class labels only     | ✅ 0.95 = excellent separation; ❌ 0.60 = poor |
| **MAE**      | Average absolute error                           | Regression with interpretable units| You want to penalize large errors more    | ✅ 2.5 = tight fit; ❌ 15.0 = loose fit     |
| **MSE**      | Average squared error                            | Regression with noisy data         | You want intuitive error units            | ✅ 6.0 = low variance; ❌ 200.0 = unstable |
| **RMSE**     | Square root of MSE                               | Regression with large error sensitivity | You want simplicity over precision    | ✅ 2.4 = precise; ❌ 14.1 = erratic         |
| **R² Score** | Proportion of variance explained                 | Overall regression model fit       | Data is non-linear or poorly scaled       | ✅ 0.92 = strong fit; ❌ 0.30 = weak model |

</details>

---

