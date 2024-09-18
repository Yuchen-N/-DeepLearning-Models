import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from flower_Dataset import read_split_data
from flower_Dataset import MyDataSet
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import torchvision.models as models
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
epochs = 20
batch_size = 16
save_steps = 10
num_workers = 4

lr = 0.001
lr_step_size = 40

data_root = "D:/Ai/viewModel/flower_photos"
train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(data_root)

img_size = 224
train_transform = transforms.Compose([transforms.RandomResizedCrop(img_size),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

val_transform = transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                      transforms.CenterCrop(img_size),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


train_dataset = MyDataSet(images_path=train_images_path,
                          images_class=train_images_label,
                          transform=train_transform)

val_dataset = MyDataSet(images_path=val_images_path,
                        images_class=val_images_label,
                        transform=val_transform)


train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=num_workers)

val_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         pin_memory=True,
                                         num_workers=num_workers)

print("len(train_loader) = {}".format(len(train_loader)))
print("len(val_loader) = {}".format(len(val_loader)))

class RunningAverage():

    def __init__(self):
        self.steps = 0
        self.loss_sum = 0
        self.acc_sum = 0
    
    def update(self, loss, acc):
        self.loss_sum += loss
        self.acc_sum += acc
        self.steps += 1
    
    def __call__(self):
        return self.loss_sum/float(self.steps), self.acc_sum/float(self.steps)
    

def train_and_evaluate(model, train_dataloader, val_dataloader, criteria, optimizer, scheduler, epochs, save_steps):
    
    best_val_acc = 0.0
    for epoch in range(epochs):
        
        print("Epoch {}/{}".format(epoch + 1, epochs))

        # ---------- train ------------
        
        model.train()
        metric_avg = RunningAverage()
        
        for i, (train_batch, labels_batch) in enumerate(train_dataloader):

            train_batch, labels_batch = train_batch.to(device), labels_batch.to(device)
            output_batch = model(train_batch)
            loss = criteria(output_batch, labels_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % save_steps == 0:
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()

                predict_labels = np.argmax(output_batch, axis=1)
                acc = np.sum(predict_labels == labels_batch) / float(labels_batch.size)

                metric_avg.update(loss.item(), acc)
            
        scheduler.step()
        train_loss, train_acc = metric_avg()
        print("- Train metrics: loss={}, acc={}".format(train_loss, train_acc))      
        
        

        # ---------- validate ------------
        model.eval()
        metric_avg = RunningAverage()

        for val_batch, labels_batch in val_dataloader:
            val_batch, labels_batch = val_batch.to(device), labels_batch.to(device)

            output_batch = model(val_batch)
            loss = criteria(output_batch, labels_batch)

            output_batch = output_batch.data.cpu().numpy()
            labels_batch = labels_batch.data.cpu().numpy()

            predict_labels = np.argmax(output_batch, axis=1)
            acc = np.sum(predict_labels == labels_batch) / float(labels_batch.size)

            metric_avg.update(loss.item(), acc)

        val_loss, val_acc = metric_avg()
        print("- Validate metrics: loss={}, acc={}".format(val_loss, val_acc))

if __name__ == '__main__':
    resnet34 = nn.Sequential(models.resnet34(),
                         nn.ReLU(),
                         nn.Dropout(0.5),
                         nn.Linear(in_features=1000, out_features=5, bias=True))
    resnet34.to(device)

    criteria = nn.CrossEntropyLoss()
    optimizer = optim.SGD(resnet34.parameters(), lr=lr, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=lr_step_size, gamma=0.1)

    train_and_evaluate(resnet34, train_loader, val_loader, criteria, optimizer, scheduler, epochs, save_steps)