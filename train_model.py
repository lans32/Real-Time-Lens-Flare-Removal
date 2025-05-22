import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
import cv2

class LensFlareDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.with_flare_dir = os.path.join(data_dir, 'with_flare')
        self.without_flare_dir = os.path.join(data_dir, 'without_flare')
        
        # Получаем список файлов и убираем префиксы для сопоставления
        self.with_flare_files = [f for f in os.listdir(self.with_flare_dir) 
                                if f.endswith(('.jpg', '.png')) and not f.startswith('.')]
        self.image_indices = [f.split('_')[1] for f in self.with_flare_files]
        
    def __len__(self):
        return len(self.with_flare_files)
    
    def __getitem__(self, idx):
        index = self.image_indices[idx]
        
        # Загружаем изображение с бликами (input_XXXXX.png)
        with_flare_path = os.path.join(self.with_flare_dir, f'input_{index}')
        with_flare = Image.open(with_flare_path).convert('RGB')
        
        # Загружаем изображение без бликов (gt_XXXXX.png)
        without_flare_path = os.path.join(self.without_flare_dir, f'gt_{index}')
        without_flare = Image.open(without_flare_path).convert('RGB')
        
        if self.transform:
            with_flare = self.transform(with_flare)
            without_flare = self.transform(without_flare)
            
        return with_flare, without_flare

class LensFlareModel(nn.Module):
    def __init__(self):
        super(LensFlareModel, self).__init__()
        
        # Энкодер
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Декодер
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            
            nn.ConvTranspose2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            
            nn.ConvTranspose2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train_model():
    # Проверяем доступность GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Создаем модель
    model = LensFlareModel().to(device)
    
    # Определяем преобразования
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    # Создаем датасет и загрузчик данных
    dataset = LensFlareDataset('data', transform=transform)
    print(f"Found {len(dataset)} image pairs")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Определяем функцию потерь и оптимизатор
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Обучаем модель
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    
    # Сохраняем модель
    torch.save(model.state_dict(), 'lens_flare_model.pth')
    
    # Экспортируем модель в ONNX
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    torch.onnx.export(model, dummy_input, 'lens_flare_model.onnx',
                     input_names=['input'],
                     output_names=['output'],
                     dynamic_axes={'input': {0: 'batch_size'},
                                 'output': {0: 'batch_size'}})

if __name__ == '__main__':
    train_model() 