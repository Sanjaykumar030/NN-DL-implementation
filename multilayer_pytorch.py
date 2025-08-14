import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

np.random.seed(1)
X_np = np.random.rand(500, 2).astype(np.float32)
Y_np = (np.sum(X_np, axis=1) > 1.0).astype(np.float32).reshape(-1, 1)  # meaningful labels

X = torch.tensor(X_np, dtype=torch.float32)
Y = torch.tensor(Y_np, dtype=torch.float32)

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

model = DNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 100
model.train()
for epoch in range(epochs):
    outputs = model(X)
    loss = criterion(outputs, Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        model.eval()
        with torch.no_grad():
            preds = (outputs > 0.5).float()
            acc = (preds == Y).float().mean()
        print(f"Epochs [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {acc*100:.2f}%")
        model.train()

with torch.no_grad():
    predictions = (model(X) > 0.5).float()
