import torch
import torch.nn as nn
import os
from mnist.model.mnist_model import MnistModel
from torch.utils.data import DataLoader
from mnist.dataset.mnist_dataset import MNISTDataset
from tqdm import tqdm


class Trainer:
    def __init__(self):
        self.device=torch.device("cpu")
        self.n_epochs=20
        self.n_classes=10
        self.model = MnistModel()
        self.model = self.model.to(self.device)
        self.lossfn = nn.CrossEntropyLoss()
        self.lr = 1e-3
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr)
        self.batch_size = 32
        self.output_dir = "checkpoints"
        self.build_dataloaders()
        self.loginfo()

    def loginfo(self):
        print(f"device : {self.device}")
        print(f"number of epochs : {self.n_epochs}")
        print(f"number of classes : {self.n_classes}")
        print(f"batch_size : {self.batch_size}")
        print(f"lenght traning dataset : {self.training_dataset.__len__()}")
        print(f"lenght test dataset : {self.test_dataset.__len__()}")
        print(f"lenght training loader : {len(self.training_loader)}")
        print(f"lenght test loader : {len(self.test_loader)}")

    def save_model(self, epoch):
        save_path = os.path.join(self.output_dir, f"model_epoch_{epoch+1}.pth")
        torch.save(self.model.state_dict(), save_path)

    def build_dataloaders(self):
        self.training_dataset = MNISTDataset(train=True)
        self.test_dataset = MNISTDataset(train=False)
        self.test_loader = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False)
        self.training_loader = DataLoader(dataset=self.training_dataset, batch_size=self.batch_size, shuffle=True)

    def train(self):
        for epoch in range(self.n_epochs):
            training_loss = self.train_one_epoch(epoch)
            validation_loss, accuracy = self.validate_one_epoch(epoch)
            print(f"epoch = {epoch+1}")
            print(f"training loss = {training_loss}")
            print(f"validation loss = {validation_loss}")
            print(f"accuracy = {accuracy*100}")
            self.save_model(epoch)
        


    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.training_loader, desc=f"training epoch {epoch+1}")

        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            predictions = self.model(images)
            loss = self.lossfn(predictions, labels)
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()

        avg_loss = total_loss / len(self.training_loader)

        return avg_loss


    def validate_one_epoch(self, epoch):
        self.model.eval()
        total_loss = 0.0
        n_correct = 0
        n_elems = 0
        pbar = tqdm(self.test_loader, desc=f"validating epoch {epoch+1}")

        with torch.no_grad():
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                predictions = self.model(images)
                loss = self.lossfn(predictions, labels)
                total_loss += loss.item()
                indices = torch.argmax(predictions, 1)
                n_correct += (indices == labels).sum().item()
                n_elems += labels.shape[0]

        avg_loss = total_loss / len(self.test_loader)
        accuracy = n_correct / n_elems

        return avg_loss, accuracy
