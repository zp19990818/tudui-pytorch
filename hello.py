# Dataset类实战

from PIL import Image
import os
from torch.utils.data import Dataset


class MyData(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)


root_dir1 = "Image"
people_label_dir = "people"
people_dataset = MyData(root_dir1, people_label_dir)


path = os.path.join(root_dir1, people_label_dir)
imagepath = os.listdir(path)
idx = 1
img_name = imagepath[idx]
img_item_path1 = os.path.join(root_dir1,people_label_dir, img_name)
img = Image.open(img_item_path1)

img.show()