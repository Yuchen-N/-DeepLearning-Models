import os 
from PIL import Image
from torch.utils.data import Dataset

class TinyImageNetDataset(Dataset):
    def __init__(self, data_dir, annotations_file, transform = None):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.imagePath = []
        self.labels = []
        self.class_to_index = {}  

        with open(annotations_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                img_name = parts[0]
                label = parts[1]
                if label not in self.class_to_index:
                    self.class_to_index[label] = len(self.class_to_index)
                label_index = self.class_to_index[label]
                images_Path = os.path.join(data_dir,"images")
                image_Path = os.path.join(images_Path,img_name)
                self.imagePath.append(image_Path)
                self.labels.append(label_index)


    def __len__(self):
        return len(self.imagePath)
    
    def __getitem__(self, index):
        image_Path = self.imagePath[index]
        label = self.labels[index]
        image = Image.open(image_Path).convert("RGB")
        
        if(self.transform):
            image = self.transform(image)
        
        return  image, label