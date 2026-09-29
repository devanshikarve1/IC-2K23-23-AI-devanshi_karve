# Lab 5 - Decision Tree Classifier and Model Evaluation

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

iris = load_iris()

X = iris.data
y = iris.target

print("=== Decision Tree Classifier ===")
print("Dataset:", "Iris Dataset")
print("Number of samples:", len(X))
print("Number of features:", X.shape[1])


# --------------------------------------------------
# 2. Train-Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 3. Create and Train Decision Tree
# --------------------------------------------------

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

print("\nDecision Tree model trained successfully.")


# --------------------------------------------------
# 4. Prediction
# --------------------------------------------------

y_pred = model.predict(X_test)

print("\nPredictions:")
print(y_pred)


# --------------------------------------------------
# 5. Model Evaluation
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n=== Model Evaluation ===")
print("Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    )
)


# --------------------------------------------------
# 6. Decision Tree Visualization
# --------------------------------------------------

plt.figure(figsize=(14, 8))

plot_tree(
    model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True
)

plt.title("Decision Tree Classifier")
plt.tight_layout()
plt.savefig("decision_tree.png")
plt.show()

print("\nDecision tree visualization saved as decision_tree.png.")
print("\nLab 5 completed successfully.")
