#%%
import argparse
import sys
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from src.models.model import MyAwesomeModel

#%%


dataset_path = Path("/data/processed/MNIST/processed")
batch_size = 64


class Train(object):
    """Helper class that will help launch class methods as commands
    from a single script
    """

    def __init__(self):
        # =============================================================================
        #         parser = argparse.ArgumentParser(
        #             description="Script for either training or evaluating",
        #             usage="python main.py <command>"
        #         )
        #         parser.add_argument("command", help="Subcommand to run")
        #         args = parser.parse_args(sys.argv[1:2])
        #         if not hasattr(self, args.command):
        #             print('Unrecognized command')
        #
        #             parser.print_help()
        #             exit(1)
        #         # use dispatch pattern to invoke method with same name
        #         getattr(self, args.command)()
        # =============================================================================
        self.train()

    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description="Training arguments")
        parser.add_argument("--lr", default=0.1)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)

        # Implement training loop
        model = MyAwesomeModel(10)
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        # train_set, test_set = mnist()

        # Data loading
        mnist_transform = transforms.Compose([transforms.ToTensor()])
        ""
        # trainset = datasets.MNIST(DATA_PATH, download=True, train=True, transform=transform)

        train_dataset = MNIST(
            dataset_path, transform=mnist_transform, train=True, download=True
        )
        train_set = DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )

        epochs = 20
        steps = 0
        train_losses = []
        for e in range(epochs):
            running_loss = 0
            for images, labels in train_set:
                log_ps = model(images)
                loss = criterion(log_ps, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                steps += 1
                train_losses.append(loss.item() / 64)
            print(f"Training loss: {running_loss/len(train_set)}")

        torch.save(model.state_dict(), "model.pth")


if __name__ == "__main__":
    Train()


# %%
