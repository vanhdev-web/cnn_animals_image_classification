from torch.utils.data import Dataset
import os
import cv2
from torchvision.transforms import ToTensor, Compose
import numpy as np
from PIL import Image

class AnimalDataset(Dataset):
    def __init__(self, root, train = True, transform = None):
        if train:
            path = os.path.join(root,"data/train" )
        else:
            path = os.path.join(root,"data/test" )

        self.categories = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']

        self.transform = transform
        self.image_files = []
        self.labels = []
        for category in self.categories:
            category_folder = os.path.join(path, category)
            for file_name in os.listdir(category_folder):
                self.labels.append(self.categories.index(category))
                self.image_files.append(os.path.join(category_folder,file_name))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        label = self.labels[item]
        image = self.image_files[item]
        # image = cv2.imread(image)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.open(image).convert("RGB")

        if self.transform:
            image = self.transform(image)
        return image, label
if __name__ == '__main__':
    root = "../Animals"
    transform = Compose([
        ToTensor()
    ])
    training_dat = AnimalDataset(root = root, train= True, transform= transform)
    image, label = training_dat.__getitem__(220)
    print(image.shape)
    print(image.dtype)
