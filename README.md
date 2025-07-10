# 🤖 Machine Learning from Data - Course Summary

<div align="center">

![Machine Learning](https://img.shields.io/badge/Machine%20Learning-CS3141-blue?style=for-the-badge)
![University](https://img.shields.io/badge/Reichman%20University-2025-orange?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-green?style=for-the-badge)

*A comprehensive overview of machine learning concepts covering supervised learning, unsupervised learning, and optimization techniques.*

**Instructors:** Prof. Ilan Gronau & Dr. Alon Kipnis

</div>

---

## 📚 Table of Contents

- [🎯 Supervised Learning](#-supervised-learning)
  - [🔍 K-Nearest Neighbors](#-k-nearest-neighbors-k-nn)
  - [📈 Polynomial Regression](#-polynomial-regression)
  - [🌳 Decision Trees](#-decision-trees)
  - [🎲 Bayesian Classification](#-bayesian-classification)
  - [🧠 Perceptron](#-perceptron)
  - [📊 Logistic Regression](#-logistic-regression)
  - [⚡ Support Vector Machines](#-support-vector-machines-svm)
  - [📏 Model Evaluation](#-model-evaluation)
- [🔍 Unsupervised Learning](#-unsupervised-learning)
  - [📊 Data Distribution Learning](#-data-distribution-learning)
  - [🔗 Clustering](#-clustering)
  - [🔄 Expectation Maximization](#-expectation-maximization)
  - [📉 Dimensionality Reduction](#-dimensionality-reduction)
- [⚙️ Optimization](#️-optimization)
  - [🎯 Gradient-Based Methods](#-gradient-based-methods)
  - [📉 Loss Functions](#-loss-functions)
  - [🔒 Constrained Optimization](#-constrained-optimization)

---

## 🎯 Supervised Learning

Supervised learning involves training models on labeled data to make predictions on new, unseen data. The training process uses input-output pairs to learn patterns and relationships.

### 🔍 K-Nearest Neighbors (K-NN)

**Core Concept:** Make predictions based on the k nearest neighbors in the feature space.

- **For Regression:** Average the values of k nearest neighbors
- **For Classification:** Take majority vote among k nearest neighbors
- **Hyperparameter:** k (number of neighbors)

**Pros:**
- ✅ Good fit to training data with minimal assumptions
- ✅ Works well with non-linear patterns

**Cons:**
- ❌ Computationally inefficient for large datasets
- ❌ Struggles with high-dimensional spaces (curse of dimensionality)

### 📈 Polynomial Regression

**Core Concept:** Fit polynomial functions to data by transforming features into higher-degree terms.

- **Process:** Transform features using polynomial basis functions → Solve linear regression
- **Hyperparameter:** k (maximum degree of polynomial)
- **Method:** Minimize sum-of-squares loss via gradient descent or pseudo-inverse

**Use Case:** When relationships between variables are non-linear but can be captured by polynomial terms.

### 🌳 Decision Trees

**Core Concept:** Create a tree-like model of decisions to classify data points.

- **Training:** Iterative node splitting using attributes that maximize impurity reduction
- **Key Metrics:** Entropy, Gini Impurity, Information Gain, Chi-squared test
- **Hyperparameters:** Max depth, max leaves, min samples per leaf, min impurity decrease

**Pros:**
- ✅ Excellent for categorical features
- ✅ No specific modeling assumptions
- ✅ Interpretable results

**Cons:**
- ❌ Can create very large, inefficient models
- ❌ Rectangular decision boundaries only
- ❌ Prone to overfitting

### 🎲 Bayesian Classification

**Core Concept:** Use Bayes' theorem and probability distributions to classify data.

- **Training:** Fit class priors and class-conditional distributions via maximum likelihood
- **Prediction:** Minimize expected risk using posterior probabilities
- **Variants:** MAP (uniform cost), ML (uniform priors), Naïve Bayes (feature independence)

**Formula:** `c(x) = argmin_j Σ π_l f_{X|Y=l}(x) λ_{j,l}`

**Pros:**
- ✅ Easily scales to multiple classes
- ✅ Flexible decision boundaries
- ✅ Can incorporate different costs for classification errors

**Cons:**
- ❌ Challenging to learn distributions in high dimensions

### 🧠 Perceptron

**Core Concept:** Linear classifier that finds a separating hyperplane.

- **Capability:** Finds linear separation boundaries
- **Extension:** Non-linear boundaries via feature mapping or dual formulation with kernels
- **Status:** Not practical (SVM is the better alternative)

### 📊 Logistic Regression

**Core Concept:** Linear classifier optimized for probabilistic interpretation.

- **Method:** Minimize binary cross-entropy (BCE) loss
- **Decision Boundary:** Linear
- **Extensions:** Natural extension to multiple classes
- **Hyperparameters:** None (typically)

**Best For:** Cases with no pure linear separation, provides probability estimates for classifications.

### ⚡ Support Vector Machines (SVM)

**Core Concept:** Find the optimal separating hyperplane with maximum margin.

- **Objective:** Maximize margin between classes
- **Extensions:** Kernel functions for non-linear boundaries
- **Slack Variables:** Allow margin violations and classification errors
- **Loss Function:** Hinge loss
- **Hyperparameters:** Kernel function, kernel parameters, C (slack variable penalty)

**Pros:**
- ✅ The "go-to" method for classification
- ✅ Flexible decision boundaries via kernels
- ✅ Some interpretability through support vectors

**Cons:**
- ❌ Relatively computationally heavy to train

### 📏 Model Evaluation

**Key Concepts:**
- **Bias-Variance Tradeoff:** Balance between underfitting and overfitting
- **Data Splitting:** Train → Validate → Test workflow
- **Metrics:** Confusion matrix, precision, recall, F1-score, ROC AUC

**Process:**
1. **Training Set:** Fit models
2. **Validation Set:** Select hyperparameters  
3. **Test Set:** Assess final generalization error

---

## 🔍 Unsupervised Learning

Unsupervised learning finds patterns and structure in data without labeled examples, focusing on discovering hidden relationships and groupings.

### 📊 Data Distribution Learning

**Core Concept:** Learn the underlying probability distribution of data.

**Methods:**
- **Non-parametric:** Histogram smoothing techniques
- **Parametric:** Maximum likelihood estimation (typically log-likelihood)

**Common Distributions:**
- Binomial, Poisson, Exponential
- **Normal Distributions:** Univariate/Multivariate Gaussians (particularly useful)
- **Gaussian Mixture Models (GMMs)**

**Applications:** Bayesian classification, understanding data properties (mean, modes, variation)

### 🔗 Clustering

**Core Concept:** Partition data into "closely clustered" subsets of samples.

**Algorithms:**
- **K-means:** Minimize within-cluster spread for given number of clusters (k)
- **Hierarchical Clustering:** Build hierarchy represented by dendrogram

**Applications:** 
- Describe data structure
- Understand generative processes (e.g., evolutionary trees)
- Data exploration and segmentation

### 🔄 Expectation Maximization

**Core Concept:** Specialized optimization for complex probabilistic models with hidden variables.

**Key Features:**
- ✅ Iterative algorithm without requiring learning rate tuning
- ✅ Useful when likelihood is simplified by introducing hidden variables
- ✅ Common for cluster labels and mixture model components

**Applications:** GMM estimation, clustering with probabilistic assignments

### 📉 Dimensionality Reduction

**Methods:** PCA (Principal Component Analysis) + LDA (Linear Discriminant Analysis)

*Note: Not covered in exam for this course year*

---

## ⚙️ Optimization

Optimization techniques are fundamental to training machine learning models, involving the minimization or maximization of objective functions.

### 🎯 Gradient-Based Methods

**Core Concept:** Use gradient information to iteratively find optimal parameters.

**Analytical Solutions:**
- Pseudoinverse method for linear regression
- Maximum likelihood estimates for many distributions

**Gradient Descent Variants:**
- **Standard GD:** Full dataset gradient computation
- **Stochastic GD:** Use batches for efficiency
- **Sub-gradient Descent:** Handle non-continuous gradient points (e.g., hinge loss)

**Key Considerations:**
- Learning rate tuning
- Initial value selection (multiple starting points for local minima)
- Convergence criteria

### 📉 Loss Functions

**Common Loss Functions:**

1. **Least Squares (Linear Regression):**
   ```
   J(θ;D) = Σ(f_θ(x^(i)) - y_i)²
   ```

2. **Binary Cross Entropy (Logistic Regression):**
   ```
   BCE(w;D) = Σ[-y^(i)log(σ(w^T x^(i))) - (1-y^(i))log(1-σ(w^T x^(i)))]
   ```

3. **Hinge Loss (SVMs):**
   ```
   L_hinge = (1/2)||w||² + (C/n)Σmax{0; 1-y^(i)(w^T x^(i) + w_0)}
   ```

### 🔒 Constrained Optimization

**Core Concept:** Optimize objectives subject to constraints using Lagrangian methods.

**Key Components:**
- **Lagrangian:** Incorporate constraints into objective function
- **Lagrange Multipliers:** Handle equality and inequality constraints
- **Dual Problem:** Alternative formulation often easier to solve
- **Complementary Slackness:** Relationship between primal and dual solutions

**SVM Example:**
- **Primal:** Minimize margin subject to classification constraints
- **Dual:** Maximize margin in terms of support vectors (enables kernel trick)

---

## 🛠️ Summary Table: Supervised Learning Methods

| Method | Feature Type | Key Concepts |
|--------|-------------|--------------|
| **K-NN** | Mostly numerical | Distance metrics, lazy learning |
| **Polynomial Regression** | Numerical | Feature transformation, overfitting control |
| **Decision Trees** | Any type | Entropy, Gini, information gain, pruning |
| **Bayesian Classifiers** | Mixed | Probability distributions, priors, posteriors |
| **Logistic Regression** | Numerical | Log-odds, BCE loss, probabilistic output |
| **SVM** | Numerical | Max-margins, kernels, slack variables |

---

<div align="center">

### 🎓 Course Completion

*This summary covers the comprehensive machine learning curriculum from CS3141 at Reichman University. Each method has its strengths and optimal use cases - the key is understanding when and how to apply them effectively.*

**Happy (and quiet) Summer! 🏖️**

---

![GitHub](https://img.shields.io/badge/GitHub-Repository-black?style=flat&logo=github)
![Markdown](https://img.shields.io/badge/Made%20with-Markdown-blue?style=flat&logo=markdown)

</div>