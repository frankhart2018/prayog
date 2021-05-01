import torch
from tqdm import tqdm
from collections import namedtuple


class Trainer:
    def __init__(self, model, loss_fn, optimizer):
        self.__model = model
        self.__loss_fn = loss_fn
        self.__optimizer = optimizer

        self.__training_stats_tuple = namedtuple("TrainingStats", ["epoch", "training_acc", "training_loss"])
        self.__training_stats = []
        
    def train(self, train_loader, epochs, device=torch.device("cpu")):
        training_size = len(train_loader.dataset)

        for epoch in range(epochs):
            current_epoch_training_acc = 0
            current_epoch_training_loss = 0

            for data in tqdm(train_loader, desc=f"Epoch: {epoch+1}"):
                input_data = data['input'].to(device)
                label = data['label'].to(device)

                output = self.__model(input_data)

                loss = self.__loss_fn(label, output)
                
                self.__optimizer.zero_grad()
                loss.backward()
                self.__optimizer.step()

                current_epoch_training_loss += loss.item()
                current_epoch_training_acc += (output == label).sum().item()

            current_epoch_training_loss /= training_size
            current_epoch_training_acc /= training_size

            current_training_stats = self.__training_stats_tuple(
                epoch=epoch+1,
                training_acc=current_epoch_training_acc,
                training_loss=current_epoch_training_loss
            )

            self.__training_stats.append(current_training_stats)

        return current_training_stats
