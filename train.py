
import torch 
from models import *
from torch.utils.data.dataloader import DataLoader
from utils.dataset import Dataset
from loss import Loss_yolov1
if __name__ == "__main__":
    darknet = Darknet('./config/yolov3.cfg')
    epochs = 10
    batch_size = 2
    train_dataloader = DataLoader(Dataset(True),batch_size=batch_size, shuffle=True)
    darknet.train()
    for epoch in range(epochs):
        for i, (img, labels) in enumerate(train_dataloader):
            res = darknet(img)
            loss = Loss_yolov1()(res, labels)
            loss.backward()
            optimzer = torch.optim.SGD(darknet.parameters(), lr=0.001, momentum=0.9)
            loss.step()
            optimzer.zero_grad()
            print('epoch--{epoch}/{epochs}, current loss is --> {loss}')
