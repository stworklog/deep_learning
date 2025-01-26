import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Generate dummy data
torch.manual_seed(42)
X = torch.rand(1000, 1)  # 1000 samples, single feature
y = 2 * X + 1  # Linear relationship

# Define a deep neural network with sigmoid activation
class DeepSigmoidNet(nn.Module):
    def __init__(self):
        super(DeepSigmoidNet, self).__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(1, 128)])
        self.hidden_layers.extend([nn.Linear(128, 128) for _ in range(9)])  # 10 layers in total
        self.output_layer = nn.Linear(128, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        return self.output_layer(x)

# Instantiate the model
model = DeepSigmoidNet()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model and log gradients
epochs = 100
losses = []
gradients = []

for epoch in range(epochs):
    # Zero the parameter gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)
    losses.append(loss.item())

    # Backward pass
    loss.backward()

    # Log the gradient norms of the first layer's weights
    gradients.append(torch.norm(model.hidden_layers[0].weight.grad).item())

    # Update weights
    optimizer.step()

# Plot the loss and gradient norms
plt.figure(figsize=(12, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Gradient norms plot
plt.subplot(1, 2, 2)
plt.plot(gradients)
plt.title('Gradient Norms (First Layer)')
plt.xlabel('Epoch')
plt.ylabel('Gradient Norm')

plt.tight_layout()
plt.show()
