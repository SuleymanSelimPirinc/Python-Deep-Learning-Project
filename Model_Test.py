import numpy as np
import librosa
import subprocess
import os
import matplotlib.pyplot as plt

# .m4a dosyasını .wav formatına dönüştürme fonksiyonu
def convert_to_wav(file_path):
    try:
        if file_path.endswith('.m4a'):
            output_path = file_path.replace('.m4a', '.wav')
            subprocess.run(['ffmpeg', '-i', file_path, output_path])
            print(f"Converted {file_path} to {output_path}")
            return output_path
        else:
            print("File is already in .wav format")
            return file_path
    except Exception as e:
        print(f"Error encountered while converting file: {file_path}, {e}")
        return None

# Yeni ses dosyasını yükleme ve MFCC özelliklerini çıkarma
def extract_features_from_new_audio(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs = np.mean(mfccs.T, axis=0)
        return mfccs
    except Exception as e:
        print(f"Error encountered while parsing file: {file_path}, {e}")
        return None

# Yeni ses dosyasının yolunu belirleyin
new_audio_file = 'xxxxxxxxx'  # Kendi ses dosyanızın yolunu buraya yazın

# Ses dosyasını .wav formatına dönüştürün
new_audio_file_wav = convert_to_wav(new_audio_file)

# Yeni ses dosyasından özellikleri çıkarın
new_audio_features = extract_features_from_new_audio(new_audio_file_wav)

if new_audio_features is not None:
    # Özellikleri modele uygun hale getirin
    new_audio_features = new_audio_features.reshape(1, -1)

    # Modeli kullanarak tahmin yapın
    predicted_probabilities = model.predict(new_audio_features)
    predicted_label = np.argmax(predicted_probabilities)

    # Tahmin sonucunu yazdırın
    class_names = ['calm', 'aggressive']
    print(f"Predicted label: {predicted_label} ({class_names[predicted_label]})")

    # Tahmin olasılıklarını grafikle gösterme
    plt.figure(figsize=(8, 6))
    plt.bar(class_names, predicted_probabilities[0])
    plt.title('Predicted Probabilities')
    plt.xlabel('Class')
    plt.ylabel('Probability')
    plt.ylim(0, 1)
    plt.show()
else:
    print("Could not extract features from the new audio file.")
