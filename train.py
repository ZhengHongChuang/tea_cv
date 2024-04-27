import torch
import os
import torch.nn as nn
from model.model import cv_tea
from utils.dataLoad import MyDataLoader
from torch.utils.tensorboard import SummaryWriter

def train():
    train_root = 'data/train'
    val_root = 'data/val'
    batch_size = 64
    epoch = 10
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = cv_tea(pretrained=True).to(device=device)
    criterion = nn.CrossEntropyLoss().to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    train_dataLoader = MyDataLoader(root=train_root,batch_size=batch_size)
    val_dataLoader = MyDataLoader(root=val_root,batch_size=batch_size)
    for epoch in range(epoch):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_dataLoader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        train_loss = running_loss / len(train_dataLoader)
        train_acc = correct / total
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Train/Accuracy', train_acc, epoch)
        # print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_dataLoader)}")


        model.eval() 
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_dataLoader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_dataLoader)
        val_acc = val_correct / val_total
        writer.add_scalar('Validation/Loss', val_loss, epoch)
        writer.add_scalar('Validation/Accuracy', val_acc, epoch)
        torch.save(model.state_dict(), 'weights/model_%d.pth' % (epoch + 1))
        # if (epoch+1) % 1 == 0:
        #     torch.save(model.state_dict(), 'weights/model_%d.pth' % (epoch + 1))
    writer.close()
if __name__=='__main__':
    train()

        









