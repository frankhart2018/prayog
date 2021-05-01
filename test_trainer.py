import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import prayog.layers as layers
import prayog.models as models
import prayog.train as train


class Sample(data.Dataset):
    def __init__(self):
        self.input_data = torch.randn(100, 3)
        self.output_data = torch.randn(100, 1)

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return {
            "input": self.input_data[idx],
            "label": self.output_data[idx],
        }

dataset = Sample()
loader = data.DataLoader(dataset=dataset, batch_size=32, shuffle=True)

model = models.Sequential(
    layers.Linear(in_features=3, out_features=10),
    layers.Linear(in_features=10, out_features=1)
)


loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

trainer = train.Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer)

stats = trainer.train(train_loader=loader, epochs=2)