import os
import glob
from tqdm import tqdm
import numpy as np
import cv2

def images_labels(dataset_path):

    images=[]
    labels=[]

    for dirpath, dirname, filenames in os.walk(dataset_path):
        for file in tqdm([f for f in filenames if (f.endswith('.png') or f.endswith('.jpg'))]):
            src=dirpath + '/' + file
            img = cv2.imread(src)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = np.resize(img,(64,64,1))
            images.append(img)

            labels.append(int(file.split('_')[4]))

    labels = np.array(labels)
    images = np.array(images)
    return images, labels