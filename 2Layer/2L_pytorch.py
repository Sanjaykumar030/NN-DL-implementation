import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Model definition

class TwoLayersNN(nn.Module):
  def __init__(self, n_x, n_h, n_y):
    super(TwoLayersNN, self).__init__()
    self.hidden = nn.Linear(n_x, n_h)
    self.output_layer = nn.Linear(n_h, n_y)
    nn.init.normal_(self.hidden.weight, std=0.01)
    nn.init.zeros_(self.hidden.bias)
    nn.init.normal_(self.output_layer.weight, std=0.01)
    nn.init.zeros_(self.output_layer.bias)


  def forward(self, X):
    A1 = torch.tanh(self.hidden(X))
    A2 = torch.sigmoid(self.output_layer(A1))
    return A2
if __name__ == "__main__":
  np.random.seed(1)
  torch.manual_seed(1)

  X = np.random.randn(2, 500).astype(np.float32)
  Y = (np.sum(X, axis=0) > 0).astype(np.float32).reshape(-1,1)

  X_torch = torch.from_numpy(X.T)
  Y_torch = torch.from_numpy(Y)

  model = TwoLayersNN(n_x=2, n_h=4, n_y=1)
  criterion = nn.BCELoss()
  optimizer = optim.SGD(model.parameters(), lr=0.01)

  for epoch in range(5000):
    optimizer.zero_grad()
    outputs = model(X_torch)
    loss = criterion(outputs, Y_torch)
    loss.backward()
    optimizer.step()

    # Printing loss every 100 epochs
    if epoch % 100 == 0:
      print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


  with torch.no_grad():
    predictions = (model(X_torch) > 0.5).float()
    accuracy = (predictions == Y_torch).sum().item() / Y_torch.shape[0] * 100
    print(f"Training accuracy: {accuracy:.2f}%")
