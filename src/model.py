import torch
import torch.nn as nn

class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)        # 128x128 → 64x64
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)        # 64x64 → 32x32
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)        # 32x32 → 16x16
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = DeepfakeDetector()
    print(model)
    
    # Test
    dummy = torch.randn(4, 1, 128, 128)
    out = model(dummy)
    print(f"\nGirdi shape:  {dummy.shape}")
    print(f"Çıktı shape: {out.shape}")
    print(f"Örnek çıktı: {out.detach()}")
    
    # Parametre sayısı
    total = sum(p.numel() for p in model.parameters())
    print(f"\nToplam parametre: {total:,}")