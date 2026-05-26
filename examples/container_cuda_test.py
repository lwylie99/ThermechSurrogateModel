import torch
from torch import amp, optim, nn
from torch.utils.data import TensorDataset

from pinns import BasicMLP

# TODO: validates that container works before trying to run actual code


device_str = 'cuda:0' if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)

print(f'is cuda available: {torch.cuda.is_available()}')
print(f'device_str: {device_str}, device: {device}')

model = BasicMLP(3, 1, 5, 128).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

print(f'model: {model}')
print(f'optim: {optimizer}')

criterion = nn.MSELoss()
scaler = amp.GradScaler(device_str)
print(f'crit: {criterion}, scaler: {scaler}')

print(f'training for one epoch with auto-scaler')
model.train()
with (amp.autocast(device.type)):  # allows use of scaler
    x = torch.tensor([[1.0,2.0,3.0]], dtype=torch.float32).to(device)
    y = torch.tensor([[4.0]], dtype=torch.float32).to(device)
    for epoch in range(20):
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)

        print(f'EPOCH: {epoch}')
        print(f'\tpred: {logits}')
        print(f'\tloss: {loss}')

        print(f'\tapplying loss using scaler...')
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

print('done')