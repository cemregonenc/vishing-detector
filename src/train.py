import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import get_dataloaders
from model import DeepfakeDetector
import os

def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Cihaz: {device}")

    train_loader, val_loader = get_dataloaders(batch_size=32)

    model = DeepfakeDetector().to(device)

    # Veri dengesizliği için ağırlık (sahte 9x fazla)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, patience=2)


    best_val_loss = float("inf")
    for epoch in range(10):
        # Eğitim
        model.train()
        train_loss, correct, total = 0, 0, 0

        for i, (mel, label) in enumerate(train_loader):
            mel, label = mel.to(device), label.to(device).unsqueeze(1)

            optimizer.zero_grad()
            out = model(mel)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = (torch.sigmoid(out) > 0.5).float()
            correct += (pred == label).sum().item()
            total += label.size(0)

            if i % 50 == 0:
                print(f"Epoch {epoch+1} | Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")

        train_acc = correct / total

        # Validasyon
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for mel, label in val_loader:
                mel, label = mel.to(device), label.to(device).unsqueeze(1)
                out = model(mel)
                loss = criterion(out, label)
                val_loss += loss.item()
                pred = (torch.sigmoid(out) > 0.5).float()
                val_correct += (pred == label).sum().item()
                val_total += label.size(0)

        val_acc = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        print(f"\nEpoch {epoch+1}/10 | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val Loss: {avg_val_loss:.4f}")

        # En iyi modeli kaydet
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "models/best_model.pth")
            print(f"Model kaydedildi!")

    print("\nEğitim tamamlandı!")

if __name__ == "__main__":
    train()