import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

# Veri yolu
TRAIN_PATH = "data/raw/LA/ASVspoof2019_LA_train/flac"
PROTOCOL_PATH = "data/raw/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"

# Protokol dosyasını oku
def load_protocol(path):
    real, fake = 0, 0
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if parts[-1] == "bonafide":
                real += 1
            else:
                fake += 1
    return real, fake

real, fake = load_protocol(PROTOCOL_PATH)
print(f"Gerçek ses: {real}")
print(f"Sahte ses:  {fake}")
print(f"Toplam:     {real + fake}")

# Örnek bir ses dosyası yükle ve görselleştir
sample_file = os.path.join(TRAIN_PATH, os.listdir(TRAIN_PATH)[0])
y, sr = librosa.load(sample_file, sr=16000)
print(f"\nÖrnek dosya: {os.listdir(TRAIN_PATH)[0]}")
print(f"Süre: {len(y)/sr:.2f} saniye")
print(f"Sample rate: {sr}")

# Mel spektrogram çiz
mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
mel_db = librosa.power_to_db(mel, ref=np.max)

plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spektrogram - Örnek Ses')
plt.tight_layout()
plt.savefig('results/sample_spectrogram.png')
print("\nSpektrogram kaydedildi: results/sample_spectrogram.png")