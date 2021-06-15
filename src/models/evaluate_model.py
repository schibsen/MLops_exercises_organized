#%%
import argparse
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from src.models.model import MyAwesomeModel

#%%


dataset_path = Path("/data/processed/MNIST/processed")
batch_size = 64


class Evaluate(object):
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

    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description="Training arguments")
        parser.add_argument("--load_model_from", default="model.pth")
        # add any additional argument that you want

        model = MyAwesomeModel(10)
        dict_ = torch.load("model.pth")
        model.load_state_dict(dict_)

        # Data loading
        mnist_transform = transforms.Compose([transforms.ToTensor()])
        ""
        # trainset = datasets.MNIST(DATA_PATH, download=True, train=True, transform=transform)

        test_dataset = MNIST(
            dataset_path, transform=mnist_transform, train=False, download=True
        )
        test_set = DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=False
        )

        accuracy = 0
        counter = 0
        # turn off gradients for the purpose of speeding up the code
        with torch.no_grad():
            for images, labels in test_set:  # with batch size 64
                ps = torch.exp(model(images))
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                counter += 1
                accuracy += torch.mean(equals.type(torch.FloatTensor))
            accuracy = accuracy / counter
            print(f"Accuracy: {accuracy.item()*100}%")


if __name__ == "__main__":
    Evaluate()
