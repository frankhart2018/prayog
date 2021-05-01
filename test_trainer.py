import torch
import torch.nn as nn
import torch.optim as optim
from torch.serialization import load
import torch.utils.data as data

import prayog.layers as layers
import prayog.models as models
import prayog.train as train


class Sample(data.Dataset):
    def __init__(self):
        self.input_data = torch.tensor([1, 2, 3, 4, 5, 9, 10, 11, 12, 13], dtype=torch.float32)
        self.output_data = torch.tensor([3, 7, 7, 9, 11, 19, 21, 23, 25, 27], dtype=torch.float32)

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return {
            "input": self.input_data[idx].unsqueeze(0),
            "label": self.output_data[idx].unsqueeze(0),
        }

dataset = Sample()
loader = data.DataLoader(dataset=dataset, batch_size=2, shuffle=True)

a = iter(loader)
b = next(a)
# print(b['input'].shape)

model = models.Sequential(
    layers.Linear(in_features=1, out_features=1),
)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

trainer = train.Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer)

stats = trainer.train(train_loader=loader, epochs=1000)
print(stats)

print(model(torch.tensor([6], dtype=torch.float32).unsqueeze(0)))