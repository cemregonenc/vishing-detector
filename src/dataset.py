import os
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset

class ASVspoofDataset(Dataset):
    def __init__(self, flac_dir, protocol_path, max_len=128):
        self.flac_dir = flac_dir
        self.max_len = max_len
        self.samples = []
        
        with open(protocol_path) as f:
            for line in f:
                parts = line.strip().split()
                filename = parts[1]
                label = 0 if parts[-1] == "bonafide" else 1
                self.samples.append((filename, label))
    
    def __len__(self):
        return len(self.samples)
    
    def get_melspectrogram(self, filepath):
        y, sr = librosa.load(filepath, sr=16000, duration=3.0)
        
        # 3 saniyelik sabit uzunluk
        target = sr * 3
        if len(y) < target:
            y = np.pad(y, (0, target - len(y)))
        else:
            y = y[:target]
        
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # Normalize et -1 ile 1 arasına
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)
        
        return torch.FloatTensor(mel_db).unsqueeze(0)  # (1, 128, 128)
    
    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        filepath = os.path.join(self.flac_dir, filename + ".flac")
        mel = self.get_melspectrogram(filepath)
        return mel, torch.tensor(label, dtype=torch.float32)


def get_dataloaders(batch_size=32):
    base = "data/raw/LA"
    
    train_dataset = ASVspoofDataset(
        flac_dir=f"{base}/ASVspoof2019_LA_train/flac",
        protocol_path=f"{base}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
    )
    
    val_dataset = ASVspoofDataset(
        flac_dir=f"{base}/ASVspoof2019_LA_dev/flac",
        protocol_path=f"{base}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    print(f"Train: {len(train_dataset)} örnek")
    print(f"Val:   {len(val_dataset)} örnek")
    
    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders(batch_size=4)
    
    # İlk batch'i test et
    batch_mel, batch_label = next(iter(train_loader))
    print(f"Batch shape: {batch_mel.shape}")
    print(f"Label shape: {batch_label.shape}")
    print(f"Örnek etiketler: {batch_label}")