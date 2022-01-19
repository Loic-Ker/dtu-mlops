"""
LFW dataloading
"""
import argparse
import time
from more_itertools import peekable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import pandas as pd

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        to_np = np.asarray(img).transpose((1, 2, 0))
        axs[0, i].imshow(np.asarray(to_np))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.savefig('batch_visu.png')

 
    
    # Define dataset
    #I had some problems with the class defined I did not use it (I did not have a clue to what to do exactly)
lfw_trans = transforms.Compose([
        transforms.RandomAffine(5, (0.1, 0.1), (0.5, 2.0)),
        transforms.ToTensor()
    ])
dataset = datasets.ImageFolder('lfw',lfw_trans)


    #dataset = LFWDataset(args.path_to_folder, lfw_trans)
    
    # Define dataloader
    # Note we need a high batch size to see an effect of using many
    # number of workers
core = [0,1,2,3,4,5,6,7,8]
deviations=[]
timings=[]
get_timing = True
if __name__ == '__main__':    
    for i in core:
        dataloader = DataLoader(dataset, batch_size=2000, shuffle=True,
                            num_workers=i)
        
        if get_timing:
            # lets do so repetitions
            res = [ ]
            for _ in range(2):
                start = time.time()
                for batch_idx, batch in enumerate(dataloader):
                    print(batch_idx)
                    if batch_idx > 2:
                        break
                end = time.time()

                res.append(end - start)
            
            res = np.array(res)
            print('Timing:', np.mean(res), '+-', np.std(res)) #error in the script here
            timings+=[np.mean(res)]
            deviations+=[np.std(res)]
    results = pd.DataFrame({'timings':timings, 'deviations':deviations})
    results.to_csv('results_data.csv')
