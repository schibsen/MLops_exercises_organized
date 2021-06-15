import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torch import nn, optim

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

os.chdir("C:/Users/annas/OneDrive/Dokumente/DTU/8_semester/3weeks_MLops/MLops_exercises_organized/")

MODEL_PATH = "src\\models\\model.pth"
DATA_PATH = "data\\processed\\"
FIGURE_PATH = "reports\\figures\\"


import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from src.models.model import MyAwesomeModel

#import plotext.plot as plx

#rm -rf ./logs/


class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
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
        %tensorboard --logdir=runs

    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.1)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        writer = SummaryWriter()

        
        
        # Implement training loop 
        model = MyAwesomeModel(10)
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        #train_set, test_set = mnist()
        transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                              ])
    
        # Download and load the training data
        train_set = datasets.MNIST(DATA_PATH, download=True, train=True, transform=transform)
        train_set = torch.utils.data.DataLoader(train_set, batch_size = 64, shuffle = True)
        #test_set = datasets.MNIST(DATA_PATH, download=True, train=True, transform=transform)
        
        
        epochs = 10
        steps = 0
        train_losses = []
        training_loss = []
        for e in range(epochs):
            running_loss = 0
            for images, labels in train_set:
                
                # training stuff
                log_ps = model(images)
                loss = criterion(log_ps, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                train_losses.append(loss.item()/64)
                
                
                # tensorboard stuff
                writer.add_scalar('Loss/train', loss.item()/64, steps)
                #writer.add_scalar('Loss/test', np.random.random(), n_iter)
                #writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
                #writer.add_scalar('Accuracy/test', np.random.random(), n_iter)

                
                steps += 1
                
            training_loss.append(running_loss/len(train_set))
            print(f"Training loss: {running_loss/len(train_set)}")
            
        #
        train_losses
        plt.plot(training_loss)
        torch.save(model.state_dict(),MODEL_PATH)
        
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--load_model_from', default=MODEL_PATH)
        # add any additional argument that you want
        
        
        
        model = MyAwesomeModel(10)
        dict_ = torch.load(MODEL_PATH)
        model.load_state_dict(dict_)
        #_, test_set = mnist()
        transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                              ])
        test_set = datasets.MNIST(DATA_PATH, download=True, train=False, transform=transform)
        test_set = torch.utils.data.DataLoader(test_set, batch_size = 64, shuffle = False)
        
        accuracy = 0
        counter = 0
        # turn off gradients for the purpose of speeding up the code
        with torch.no_grad():
            for images, labels in test_set: # with batch size 64
                ps = torch.exp(model(images))
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                counter += 1
                accuracy += torch.mean(equals.type(torch.FloatTensor))
            accuracy = accuracy / counter
            print(f'Accuracy: {accuracy.item()*100}%')

if __name__ == '__main__':
    TrainOREvaluate()
    
    
    
    
    
    
    
    
    