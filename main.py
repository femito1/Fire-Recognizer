import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from d2l import torch as d2l
class TestDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_paths = [
            os.path.join(root, fname)
            for fname in os.listdir(root)
            if fname.endswith(('jpg', 'png', 'jpeg'))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_path
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(
        size=224,
        scale=(0.85, 1.15),
        ratio=(0.75, 1.33)
    ),
    transforms.ToTensor(),
])

trainset = datasets.ImageFolder(
    root = 'dl2425_challenge_dataset/train',
    transform = transform
)
valset = datasets.ImageFolder(
    root = 'dl2425_challenge_dataset/val',
    transform = transform
)
testset = TestDataset(
    root = 'dl2425_challenge_dataset/test',
    transform = transform
)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
valloader = DataLoader(valset, batch_size=64, shuffle=False)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

def init_cnn(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)

class NandNet(d2l.Classifier):

    def __init__(self, numchannels, lr=0.1, num_classes=2):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.LazyConv2d(numchannels, kernel_size=3, stride=1, padding=1),
            nn.LazyBatchNorm2d(), nn.ReLu(),
            nn.LazyConv2d(numchannels, kernel_size=3, stride=1, padding=1),
            nn.LazyBatchNorm2d(), nn.ReLu(),
            nn.LazyConv2d(numchannels, kernel_size=3, stride=1, padding=1),
            nn.LazyBatchNorm2d(), nn.ReLu(),
            nn.Flattten(), nn.LazyLinear(64), nn.LazyBatchNorm1d(),
            nn.ReLU(), nn.LazyLinear(16), nn.LazyBatchNorm1d(),
            nn.ReLU(), nn.LazyLinear(num_classes)
            )

trainer = d2l.Trainer(max_epochs=3, num_gpus=1)
model.apply(init_cnn)

data = {
    'train': trainloader,
    'val': valloader
}
trainer.fit(model, data)


