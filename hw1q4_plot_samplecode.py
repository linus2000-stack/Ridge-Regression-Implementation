import numpy as np
import matplotlib.pyplot as plt

def ridge_regression(tX, tY, l):
    """
    Perform ridge regression to find the optimal weight vector theta.
    
    Inputs:
    tX : numpy array
        Training features.
    tY : numpy array
        Training responses.
    l : float
        Regularization parameter λ.
        
    Outputs:
    theta : numpy array
        The optimal weight vector θ.
    """
    
    n, d = tX.shape
    I = np.eye(d)
    theta = np.linalg.inv(n * l * I + tX.T @ tX) @ tX.T @ tY
    return theta

def compute_loss(X, Y, theta):
    """Compute mean squared error loss."""
    predictions = X @ theta
    return np.mean((predictions - Y) ** 2)

# Use the first 40 entries for training and last 10 for validation
tX = np.genfromtxt('hw1_ridge_x.dat', delimiter=',')
tY = np.genfromtxt('hw1_ridge_y.dat', delimiter=',')
train_features = tX[:40]
train_responses = tY[:40]
validation_features = tX[40:]
validation_responses = tY[40:]

# Compute the result for λ = 0.15
lambda_value = 0.15
theta = ridge_regression(train_features, train_responses, lambda_value)
print(f"Theta for λ = {lambda_value} is {theta}")

# Generate values for lambda
lambda_values = np.logspace(-5, 0, 100)

training_losses = []
validation_losses = []

for l in lambda_values:
    theta = ridge_regression(train_features, train_responses, l)
    training_loss = compute_loss(train_features, train_responses, theta)
    validation_loss = compute_loss(validation_features, validation_responses, theta)
    
    training_losses.append(training_loss)
    validation_losses.append(validation_loss)

# Find the lambda that minimizes the validation loss
optimal_lambda = lambda_values[np.argmin(validation_losses)]
print(f"Optimal lambda: {optimal_lambda}")

# Plot the results
plt.plot(lambda_values, training_losses, label='Training Loss')
plt.plot(lambda_values, validation_losses, label='Validation Loss')
plt.xscale('log')
plt.xlabel('Lambda')
plt.ylabel('Loss')
plt.legend()
plt.show()

