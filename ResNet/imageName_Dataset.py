import torch.nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from torchvision.transforms import transforms
import os

class customDataset(Dataset):
    def __init__(self, dataFolder, transform = None):
        super().__init__()
        self.dataPath = dataFolder
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_index = {} #类别转换成索引的映射

        for idx, class_name in enumerate(os.listdir(dataFolder)):
            class_dir = os.path.join(dataFolder, class_name)
            images_dir = os.path.join(class_dir, "images")
            if os.path.isdir(images_dir):
                self.class_to_index[class_name] = idx # class1name :0 class2name : ... ...1
                for img_name in os.listdir(images_dir):
                    img_path = os.path.join(images_dir, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(idx)
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        img_path = self.image_paths[index]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)
        
        return image, label
