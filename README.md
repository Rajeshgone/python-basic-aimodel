# python-basic-aimodel
python-basic-aimodel
# ================================================
# PYTHON BASIC AI MODEL - Simple Neural Network
# (From scratch using only NumPy)
# ================================================

import numpy as np

# ======================
# 1. Activation Functions
# ======================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# ======================
# 2. Basic Neural Network Class
# ======================
class BasicAIModel:
    def __init__(self, input_size, hidden_size, output_size):
        # Random weight initialization
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.weights2 = np.random.randn(hidden_size, output_size)
        self.bias1 = np.zeros((1, hidden_size))
        self.bias2 = np.zeros((1, output_size))
        
    def forward(self, X):
        """Forward propagation"""
        self.layer1 = np.dot(X, self.weights1) + self.bias1
        self.activation1 = sigmoid(self.layer1)
        
        self.layer2 = np.dot(self.activation1, self.weights2) + self.bias2
        self.output = sigmoid(self.layer2)
        return self.output
    
    def train(self, X, y, epochs=1000, learning_rate=0.1):
        """Train the model using backpropagation"""
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Calculate error
            error = y - output
            
            # Backpropagation
            d_output = error * sigmoid_derivative(output)
            d_hidden = np.dot(d_output, self.weights2.T) * sigmoid_derivative(self.activation1)
            
            # Update weights and biases
            self.weights2 += learning_rate * np.dot(self.activation1.T, d_output)
            self.weights1 += learning_rate * np.dot(X.T, d_hidden)
            self.bias2 += learning_rate * np.sum(d_output, axis=0, keepdims=True)
            self.bias1 += learning_rate * np.sum(d_hidden, axis=0, keepdims=True)
            
            if epoch % 100 == 0:
                loss = np.mean(np.square(error))
                print(f"Epoch {epoch} - Loss: {loss:.4f}")

# ======================
# 3. Example Usage: XOR Problem (Classic AI Test)
# ======================
if __name__ == "__main__":
    print("🚀 Training Basic AI Model on XOR Problem...\n")
    
    # XOR input and output
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    
    y = np.array([[0],
                  [1],
                  [1],
                  [0]])
    
    # Create and train model
    model = BasicAIModel(input_size=2, hidden_size=4, output_size=1)
    model.train(X, y, epochs=5000, learning_rate=0.5)
    
    print("\n✅ Training Complete!")
    print("Predictions:")
    for i in range(len(X)):
        pred = model.forward(X[i:i+1])
        print(f"Input: {X[i]} → Predicted: {pred[0][0]:.4f} → Actual: {y[i][0]}")
