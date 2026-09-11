from torch.utils.data import Dataset

from torchvision import datasets, transforms

class MNISTDataset(Dataset):
    def __init__(self, root="data", train=True):
        self.data = datasets.MNIST(
            root=root,
            train=train,
            download=True,
            transform=transforms.ToTensor()
        )

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        image, label = self.data[idx]
        return image, label