import sys
import argparse

import torch
from torch import nn, optim
import torch.nn.functional as F
import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

os.chdir("C:/Users/annas/OneDrive/Dokumente/DTU/8_semester/3weeks_MLops/MLops_exercises_organized/")

MODEL_PATH = "src\\models\\model.pth"
DATA_PATH = "data\\processed\\"
FIGURE_PATH = "reports\\figures\\"


from torchvision import datasets, transforms
from src.models.model import MyAwesomeModel
import numpy as np



from sklearn.manifold import TSNE


import matplotlib.pyplot as plt
#import plotext.plot as plx


class main():

    def __init__(self):
        self.t_SNE()
            
    
    def t_SNE(self):
        
        model = MyAwesomeModel(10)
        dict_ = torch.load(MODEL_PATH)
        model.load_state_dict(dict_)
        
        transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                              ])
        test_set = datasets.MNIST(DATA_PATH, download=True, train=False, transform=transform)
        test_set = torch.utils.data.DataLoader(test_set, batch_size = 1, shuffle = False)
        
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        
        
        features_all = []
        labels_all = []
        with torch.no_grad():
            for images, labels in test_set: # with batch size 64
                features = model(images, return_features = True)
                features_all.append(features.numpy())
                labels_all.append(labels.numpy()[0])
                
        
        #features_embedded = TSNE.fit_transform(np.stack(features_all, axis = 0))
        #print(features_embedded)
        return features_all, labels_all
        
    

#%%

features_all, labels_all = main().t_SNE()
#%%
features_np = np.stack(features_all,axis = 0).squeeze()

tsne = TSNE(n_components=2, init='pca', random_state=0)

out = tsne.fit_transform(features_np)


#%%%
#colors = plt.cm.rainbow(np.linspace(0, 1, 10))

#c_dict = zip(colors, range(9))

colors = ['tab:blue', 'tab:orange', 'tab:green','tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

c_dict = dict(zip(list(range(10)), colors))


#%%


fig = plt.figure(figsize=(15, 8))
#ax = fig.add_subplot(2, 5, 10)
colors = plt.cm.rainbow(np.linspace(0, 1, 10000))
#c_dict = zip(colors, labels_all.uniqie())


for idx, label in enumerate(zip(out, labels_all)):
    #print(idx)
    x = label[0]
    plt.scatter(x[0], x[1], c=c_dict[label[1]])
#plt.scatter(out[:,0], out[:,1], c =colors, cmap=plt.cm.rainbow)
#plt.title("t-SNE (%.2g sec)" % (t1 - t0))
#ax.xaxis.set_major_formatter(NullFormatter())
#ax.yaxis.set_major_formatter(NullFormatter())
#plt.axis('tight')
