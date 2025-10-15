# 🔍 Logistic Regression Deep Dive — From Sigmoid to Accuracy  
> **“When probabilities meet decisions, math gets strategic.”** 😅📊

This page unpacks the mechanics behind logistic regression — from sigmoid curves to classification thresholds, logit functions, and evaluation metrics.

---

<details>
<summary>📐 Sigmoid Function — The Probability Gate</summary>

### 🎯 What It Does  
Transforms any real-valued number into a probability between 0 and 1.  
The **variance** of the sigmoid output reflects how uncertain the prediction is — highest near 0.5, lowest near 0 or 1.

---

### 🧮 Sigmoid Variance Equation

$$
\sigma^2(a) = \frac{1}{(1 + e^{-a})^2}, \quad \text{where } a = -\beta_0 - \beta_1 x_1 + \dots + \beta_n x_n
$$

---

### 📐 Annotated Terms in Sigmoid Variance Equation

### 📐 Annotated Terms in Sigmoid Variance Equation

| Symbol                          | Meaning                                                                 |
|----------------------------------|-------------------------------------------------------------------------|
| \(\sigma^2(a)\)                 | Variance of the sigmoid output — reflects uncertainty in prediction     |
| \(e^{-a}\)                      | Exponential decay — controls the steepness of the sigmoid curve         |
| \(a\)                           | Linear combination of inputs and weights — raw model score              |
| \(\beta_0\)                     | Intercept term — baseline bias                                          |
| \(\beta_1, \dots, \beta_n\)     | Coefficients — influence of each feature on the prediction              |
| \(x_1, \dots, x_n\)             | Input features — the data used to make predictions                      |


---

### ☕ Intuition  
As \(a\) increases, the sigmoid output approaches 1 and variance drops — Kanak’s confidence rises.  
As \(a\) nears 0, the output hovers around 0.5 — uncertainty peaks, like debugging pre-chai.

</details>


---

<details>
<summary>🔢 Probabilities, Classes & Thresholds</summary>

### 🧠 How It Works  
Logistic regression outputs a probability \( \hat{p} \) for class membership.  
We convert this into a class label using a **threshold** (default = 0.5):

- If \( \hat{p} > \text{threshold} \) → Class 1 (e.g., “Stressed”)  
- If \( \hat{p} < \text{threshold} \) → Class 0 (e.g., “Calm”)

### ⚖️ Implications of Changing the Threshold  
| Threshold ↑ | Threshold ↓ |
|-------------|-------------|
| Fewer positives predicted | More positives predicted |
| Higher precision | Higher recall |
| May miss true positives | May include false positives |

</details>

---

<details>
<summary>📈 Decision Boundary</summary>

### 🎯 What It Is  
The point where the predicted probability crosses the threshold — typically where \( \hat{p} = 0.5 \).  
In 2D, it’s a line; in higher dimensions, it’s a hyperplane.

### ☕ Analogy  
It’s like Kanak’s debugging threshold — below 0.5, it’s “probably fine”; above 0.5, it’s “time to refactor.”

</details>

---

<details>
<summary>📊 Logit Function — Why It Matters</summary>

### 🧠 Definition  
The **logit** is the inverse of the sigmoid:



\[
\text{logit}(p) = \log\left(\frac{p}{1 - p}\right)
\]



### 🎯 Why Use It  
- Converts probabilities to **log-odds**  
- Makes the model **linear in parameters**  
- Enables interpretation of coefficients

</details>

---

<details>
<summary>🎲 Odds — The Intuition</summary>

### 🧠 What Are Odds?  
Odds represent the ratio of success to failure:



\[
\text{Odds} = \frac{p}{1 - p}
\]



### 📊 Example  
- \( p = 0.8 \) → Odds = 4 (4 times more likely to be class 1)  
- \( p = 0.2 \) → Odds = 0.25 (less likely to be class 1)

</details>

---

<details>
<summary>🚦 Classifications & Misclassifications</summary>

### ✅ True Positives (TP)  
Correctly predicted class 1

### ❌ False Positives (FP)  
Predicted class 1, but it’s actually class 0

### ❌ False Negatives (FN)  
Predicted class 0, but it’s actually class 1

### ✅ True Negatives (TN)  
Correctly predicted class 0

</details>

---

<details>
<summary>📏 Classification Accuracy</summary>

### 🧮 Formula  


\[
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
\]



### ⚠️ Caveat  
Accuracy can be misleading with **imbalanced classes** — use **precision**, **recall**, and **F1 score** for deeper insight.

</details>

---

<details>
<summary>🎯 Class of Interest</summary>

### 🧠 What It Means  
The class you care most about — often the **positive class** (e.g., “Stressed”, “Churn”, “Disease”).

### 📊 Why It Matters  
- Guides threshold tuning  
- Impacts metric selection  
- Shapes model interpretation

</details>

---

### 🎬 What’s Next?

- 📈 Visualize sigmoid curves and decision boundaries  
- ✂️ Tune thresholds for precision vs recall trade-offs  
- 🧠 Extend to **multiclass and multilabel** setups  
- ☕ Build a chai-themed classifier: “Will Kanak need caffeine?”

---

<details>
<summary>📈 Sigmoid Curve — Probability vs Raw Score</summary>

### 🎯 What It Shows  
The sigmoid function maps any input \( a \) to a probability between 0 and 1:



\[
\sigma(a) = \frac{1}{1 + e^{-a}}
\]



### 📊 Plot Characteristics  
- S-shaped curve  
- Centered at \( a = 0 \) → probability = 0.5  
- As \( a \to +\infty \), \( \sigma(a) \to 1 \)  
- As \( a \to -\infty \), \( \sigma(a) \to 0 \)

### ☕ Intuition  
Think of it as Kanak’s confidence meter — low \( a \) = uncertain, high \( a \) = confident prediction.

### 🧑‍💻 Python Snippet

```python
import numpy as np
import matplotlib.pyplot as plt

a = np.linspace(-10, 10, 100)
sigmoid = 1 / (1 + np.exp(-a))

plt.plot(a, sigmoid, label='Sigmoid Curve')
plt.axhline(0.5, color='gray', linestyle='--', label='Threshold = 0.5')
plt.xlabel('Raw Score (a)')
plt.ylabel('Probability')
plt.title('Sigmoid Function')
plt.legend()
plt.grid(True)
plt.show()
```
</details>


---

```markdown
<details>
<summary>🧭 Decision Boundary — Classification Threshold</summary>

### 🎯 What It Means  
The **decision boundary** is the point where the predicted probability crosses the classification threshold (usually 0.5).

### 📐 In 1D  
- If \( \sigma(a) > 0.5 \) → Class 1  
- If \( \sigma(a) < 0.5 \) → Class 0

### 📐 In 2D  
- The boundary becomes a **line** separating two regions  
- In higher dimensions, it’s a **hyperplane**

### 🧑‍💻 Python Snippet (2D Visualization)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1)
model = LogisticRegression()
model.fit(X, y)

# Plot decision boundary
coef = model.coef_[0]
intercept = model.intercept_
x_vals = np.linspace(X[:,0].min(), X[:,0].max(), 100)
y_vals = -(coef[0] * x_vals + intercept) / coef[1]

plt.scatter(X[:,0], X[:,1], c=y, cmap='bwr', alpha=0.7)
plt.plot(x_vals, y_vals, 'k--', label='Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Decision Boundary')
plt.legend()
plt.grid(True)
plt.show()

/<details>
