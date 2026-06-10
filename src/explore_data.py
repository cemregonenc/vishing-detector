import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Grafik teması
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 13})

TRAIN_PATH = "data/raw/LA/ASVspoof2019_LA_train/flac"
PROTOCOL_PATH = "data/raw/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"

def analyze_and_plot_data():
    print("📊 Veri bilimi ve korelasyon analizi başlıyor...")
    
    real_mins, real_maxs, real_means = [], [], []
    fake_mins, fake_maxs, fake_means = [], [], []
    
    # Korelasyon matrisi için frekans bantlarını toplayacağımız listeler
    real_mel_samples = []
    fake_mel_samples = []
    
    real_count, fake_count = 0, 0
    max_samples = 100  # İstatistiksel kararlılık için örnek sayısı
    
    with open(PROTOCOL_PATH) as f:
        for line in f:
            parts = line.strip().split()
            filename = parts[1]
            label = parts[-1]
            filepath = os.path.join(TRAIN_PATH, filename + ".flac")
            
            if not os.path.exists(filepath):
                continue
                
            # Sesi yükle ve spektrogramı hesapla (Sabit ref=1.0 ile)
            y, sr = librosa.load(filepath, sr=16000, duration=3.0)
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
            mel_db = librosa.power_to_db(mel, ref=1.0)
            
            # Zaman ekseninde ortalama alarak 128 boyutlu bir frekans vektörü elde edelim
            mel_mean_vector = mel_db.mean(axis=1)
            
            if label == "bonafide" and real_count < max_samples:
                real_mins.append(mel_db.min())
                real_maxs.append(mel_db.max())
                real_means.append(mel_db.mean())
                real_mel_samples.append(mel_mean_vector)
                real_count += 1
            elif label != "bonafide" and fake_count < max_samples:
                fake_mins.append(mel_db.min())
                fake_maxs.append(mel_db.max())
                fake_means.append(mel_db.mean())
                fake_mel_samples.append(mel_mean_vector)
                fake_count += 1
                
            if real_count >= max_samples and fake_count >= max_samples:
                break

    print("✅ Veriler toplandı. Grafikler oluşturuluyor...")
    os.makedirs('results', exist_ok=True)

    # ---------------------------------------------------------
    # GRAFİK 1: İstatiksel Dağılım Grafiği (Density Plot)
    # ---------------------------------------------------------
    fig1, axes1 = plt.subplots(1, 3, figsize=(16, 5))
    
    sns.kdeplot(real_means, color="#2ecc71", fill=True, label="Gerçek (Bonafide)", lw=2, ax=axes1[0])
    sns.kdeplot(fake_means, color="#e74c3c", fill=True, label="Sahte (Spoof)", lw=2, ax=axes1[0])
    axes1[0].set_title("Ortalama dB Dağılımı")
    axes1[0].set_xlabel("Desibel (dB)")
    axes1[0].legend()

    sns.kdeplot(real_mins, color="#2ecc71", fill=True, ax=axes1[1])
    sns.kdeplot(fake_mins, color="#e74c3c", fill=True, ax=axes1[1])
    axes1[1].set_title("Minimum dB Dağılımı (Gürültü Tabanı)")
    axes1[1].set_xlabel("Desibel (dB)")

    sns.kdeplot(real_maxs, color="#2ecc71", fill=True, ax=axes1[2])
    sns.kdeplot(fake_maxs, color="#e74c3c", fill=True, ax=axes1[2])
    axes1[2].set_title("Maksimum dB Dağılımı (Tepe Enerjisi)")
    axes1[2].set_xlabel("Desibel (dB)")
    
    plt.tight_layout()
    plt.savefig('results/veri_bilimi_db_dagilimi.png', dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # GRAFİK 2: Kutu Grafiği (Box Plot)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    data_to_plot = [real_means, fake_means, real_mins, fake_mins]
    labels = ['Gerçek Ort.', 'Sahte Ort.', 'Gerçek Min', 'Sahte Min']
    sns.boxplot(data=data_to_plot, palette=["#2ecc71", "#e74c3c", "#27ae60", "#c0392b"])
    plt.xticks(range(4), labels)
    plt.ylabel("Desibel (dB)")
    plt.title("Gerçek ve Sahte Seslerin Enerji Aralık Karşılaştırması")
    plt.savefig('results/veri_bilimi_box_plot.png', dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # GRAFİK 3: Korelasyon Matrisi (Correlation Matrix)
    # ---------------------------------------------------------
    # 128 Mel bandından analizi kolaylaştırmak adına 16 alt bant seçelim (Örnekleme)
    step = 128 // 16
    columns_labels = [f"Mel_{i*step}" for i in range(16)]
    
    real_df = pd.DataFrame(real_mel_samples).iloc[:, ::step].iloc[:, :16]
    real_df.columns = columns_labels
    real_corr = real_df.corr()

    fake_df = pd.DataFrame(fake_mel_samples).iloc[:, ::step].iloc[:, :16]
    fake_df.columns = columns_labels
    fake_corr = fake_df.corr()

    fig3, axes3 = plt.subplots(1, 2, figsize=(16, 7))
    
    # Gerçek Sesler Korelasyon Isı Haritası
    sns.heatmap(real_corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=axes3[0], cbar=False)
    axes3[0].set_title("Gerçek Sesler Frekans Korelasyonu")
    
    # Sahte Sesler Korelasyon Isı Haritası
    sns.heatmap(fake_corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=axes3[1])
    axes3[1].set_title("Sahte Sesler Frekans Korelasyonu")
    
    plt.tight_layout()
    plt.savefig('results/veri_bilimi_korelasyon_matrisi.png', dpi=300)
    plt.close()
    
    print("💾 Tüm grafikler 'results/' klasörüne başarıyla kaydedildi:")
    print("  - results/veri_bilimi_db_dagilimi.png")
    print("  - results/veri_bilimi_box_plot.png")
    print("  - results/veri_bilimi_korelasyon_matrisi.png")

if __name__ == "__main__":
    analyze_and_plot_data()
