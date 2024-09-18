import torch
from torchvision.transforms import transforms
from AlexModel import Alexnet
from imageFolder_Dataset import customDataset
from txt_Dataset import TinyImageNetDataset
from torch.utils.data import DataLoader
import argparse
from sklearn.metrics import precision_recall_curve, average_precision_score
import numpy as np

def evaluate(model, dataloader, device):
    model.eval()  # 将模型切换到评估模式
    correct = 0
    total = 0
    with torch.no_grad():  # 在评估时，不需要计算梯度
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)  
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)  # 获取预测类别
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # 计算正确的预测个数

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy:.2f}%')

def main():
    parser = argparse.ArgumentParser(description="AlexNet Evaluation parser")
    parser.add_argument("--classes_num", type=int, required=True, help="Enter the total classes in the Dataset")
    parser.add_argument("--dataPath", required=True, help="Path to the test dataset")
    parser.add_argument("--weights", required=True, help="Path to the trained model weights (.pth)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 初始化模型并加载训练好的权重
    model = Alexnet(args.classes_num).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))

    # 定义数据转换
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    annotations_file = "D:/Ai/viewModel/AlexNet/tiny-imagenet-200/val/val_annotations.txt"
    # 加载测试数据集
    test_dataset = TinyImageNetDataset(args.dataPath,annotations_file, transform=transform)
    test_dataLoader = DataLoader(test_dataset, batch_size=20, shuffle=False, num_workers=2)

    # 评估模型
    evaluate(model, test_dataLoader, device)

if __name__ == '__main__':
    main()
