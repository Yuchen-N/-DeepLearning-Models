import torch.nn as nn
import torch.optim as optim
import torch
from torchvision.transforms import transforms
from imageName_Dataset import customDataset
from txt_Dataset import TinyImageNetDataset
from ResNetModel import ResNet
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from ResNetModel import BasicBlock

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = ResNet(BasicBlock, [2, 2, 2, 2]).to(device)  # 例如使用 ResNet-18

    optimizer = optim.SGD(model.parameters(), lr = 0.4, momentum= 0.9, weight_decay= 5e-4)

    criterion = nn.CrossEntropyLoss()

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.7),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    train_dataset = customDataset("D:/Ai/viewModel/tiny-imagenet-200/train",transform)

    train_loader = DataLoader(train_dataset, batch_size= 20, shuffle=True, collate_fn= None, num_workers= 2)

    num_epochs = 30

    for epoch in range(num_epochs):
        for image, label in train_loader:
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            pred = model(image)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epoch}, Loss: {loss.item()}")
        torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')  # 保存每个 epoch 的权重

if __name__ == '__main__':
    main()