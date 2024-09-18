import torch.nn as nn
import torch
from torchvision.transforms import transforms
from AlexModel import Alexnet
from imageFolder_Dataset import customDataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    parser = argparse.ArgumentParser(description= "AlexNet Net parser")
    parser.add_argument("--classes_num", type= int, required= True, help= "Enter the total classes in the Dataset")
    parser.add_argument("--epochs", type= int, required= True, help="Enter the total training epochs")
    parser.add_argument("--dataPath", required= True, help= "Dataset Path needed for dataset")
    args = parser.parse_args()

    model = Alexnet(args.classes_num).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9, weight_decay= 5e-4)

    transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.7),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])

    train_dataset = customDataset(args.dataPath,transform=transform)

    train_dataLoader = DataLoader(train_dataset, batch_size= 20, shuffle= True, collate_fn=None, num_workers= 2)

    num_epoch = args.epochs

    for epoch in range(num_epoch):
        model.train()
        for image, label in train_dataLoader:
            image, label = image.to(device), label.to(device)
            optimizer.zero_grad()
            prediction = model(image)
            loss = criterion(prediction, label)
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch+1}/{num_epoch}, Loss: {loss.item()}")
        torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')  # 保存每个 epoch 的权重

if __name__ == '__main__':
    main()