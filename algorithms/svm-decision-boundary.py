import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                           n_informative=2, n_clusters_per_class=1,
                           class_sep=2.5, random_state=0)
y = np.where(y == 0, -1, 1) 

X_b = np.hstack((X, np.ones((X.shape[0], 1))))

epochs = 1000
learning_rate = 0.001
lambda_param = 0.01

w = np.zeros(X_b.shape[1])
losses = []

for epoch in range(epochs):
    loss = 0
    for i in range(X.shape[0]):
        condition = y[i] * np.dot(X_b[i], w)
        if condition >= 1:
            grad = lambda_param * w
            loss += 0
        else:
            grad = lambda_param * w - y[i] * X_b[i]
            loss += 1 - condition
        w -= learning_rate * grad
    total_loss = loss + (lambda_param / 2) * np.dot(w, w)

    losses.append(total_loss)

plt.figure(figsize=(8, 5))
plt.plot(range(epochs), losses, label="Hinge Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Hinge Loss vs Epochs (Gradient Descent)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


def plot_decision_boundary_with_margins(X, y, w):
    plt.figure(figsize=(8, 6))

    colors = ['red' if label == 1 else 'blue' for label in y]
    plt.scatter(X[:, 0], X[:, 1], c=colors, label='Data', edgecolors='k')

    x0 = np.linspace(np.min(X[:, 0]) - 1, np.max(X[:, 0]) + 1, 200)
    x1 = -(w[0] * x0 + w[2]) / w[1] 
    x1_margin_up = -(w[0] * x0 + w[2] - 1) / w[1]
    x1_margin_down = -(w[0] * x0 + w[2] + 1) / w[1]

    plt.plot(x0, x1, 'k-', label='Decision Boundary')
    plt.plot(x0, x1_margin_up, 'k--', label='Margin')
    plt.plot(x0, x1_margin_down, 'k--')

    support_vectors = []
    for i in range(len(X)):
        if y[i] * np.dot(np.append(X[i], 1), w) < 1.05: 
            support_vectors.append(X[i])

    support_vectors = np.array(support_vectors)
    if len(support_vectors) > 0:
        plt.scatter(support_vectors[:, 0], support_vectors[:, 1],
                    facecolors='none', edgecolors='green',
                    s=100, label='Support Vectors')

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("SVM Decision Boundary with Margins and Support Vectors (Gradient Descent)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_decision_boundary_with_margins(X, y, w)