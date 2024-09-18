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

epochs = 20
batch_size = 16
save_steps = 10
num_workers = 4
lr = 0.001
lr_step = 40
alpha = 0.7
temperature = 7 # softmax函数的温度T

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def loss_fn_kd(student_outputs, teacher_outputs, labels, alpha, temperature):

    alpha = alpha
    T = temperature

    KD_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(student_outputs/T, dim=1),
                                                  F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(student_outputs, labels) * (1.- alpha)

    return KD_loss

def kd_train_and_evaluate(teacher_model, student_model, train_dataloader, val_dataloader, criteria, optimizer, scheduler, alpha, temperature):
    
    best_val_acc = 0.0
    for epoch in range(epochs):
        
        print("Epoch {}/{}".format(epoch + 1, epochs))

        # ---------- train ------------
        
        student_model.train()
        metric_avg = RunningAverage()
        
        for i, (train_batch, labels_batch) in enumerate(train_dataloader):
            train_batch, labels_batch = train_batch.to(device), labels_batch.to(device)
            
            student_outputs = student_model(train_batch)
            
            with torch.no_grad():
                teacher_outputs = teacher_model(train_batch)
                
            loss = criteria(student_outputs, teacher_outputs, labels_batch, alpha, temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % save_steps == 0:
                student_outputs = student_outputs.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()

                predict_labels = np.argmax(student_outputs, axis=1)
                acc = np.sum(predict_labels == labels_batch) / float(labels_batch.size)
                metric_avg.update(loss.item(), acc)
            
        scheduler.step()
        train_loss, train_acc = metric_avg()
        print("- Train metrics: loss={}, acc={}".format(train_loss, train_acc))      

        # ---------- validate ------------

        student_model.eval()
        metric_avg = RunningAverage()

        for val_batch, labels_batch in val_dataloader:
            val_batch, labels_batch = val_batch.to(device), labels_batch.to(device)

            student_outputs = student_model(val_batch)
            loss = 0
            
            student_outputs = student_outputs.data.cpu().numpy()
            labels_batch = labels_batch.data.cpu().numpy()

            predict_labels = np.argmax(student_outputs, axis=1)
            acc = np.sum(predict_labels == labels_batch) / float(labels_batch.size)
            metric_avg.update(loss, acc)
        _, val_acc = metric_avg()
        print("- Validate metrics: acc={}".format(val_acc))


if __name__ == '__main__':
    teacher_model = nn.Sequential(models.resnet50(),
                                nn.Dropout(0.5),
                                nn.ReLU(),
                                nn.Linear(in_features=1000, out_features=5, bias=True))

    checkpoint = torch.load('D:/Ai/viewModel/Distillation_Teacher/pretrained_tm_weights/best.pth')
    teacher_model.load_state_dict(checkpoint["state_dict"])
    teacher_model.to(device)

    print("prepare the teacher model --- done")

    student_model = nn.Sequential(models.resnet34(),
                            nn.ReLU(),
                            nn.Dropout(0.5),
                            nn.Linear(in_features=1000, out_features=5, bias=True))

    student_model.to(device)
    print("prepare the student model --- done")

    optimizer = optim.SGD(student_model.parameters(), lr=lr, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=lr_step, gamma=0.1)

    kd_train_and_evaluate(teacher_model, student_model, train_loader, val_loader, loss_fn_kd, optimizer, scheduler, alpha, temperature)