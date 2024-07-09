import matplotlib.pyplot as plt
import numpy as np
import os
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pickle
import librosa.display

# Özellik çıkarımı fonksiyonu
def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs = np.mean(mfccs.T, axis=0)
        return mfccs
    except Exception as e:
        print(f"Error encountered while parsing file: {file_path}, {e}")
        return None

# Veri augmentasyonu fonksiyonları
def add_noise(data, noise_factor=0.005):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    return augmented_data

def shift(data, shift_max=0.2):
    shift = np.random.randint(len(data) * shift_max)
    if np.random.rand() > 0.5:
        shift = -shift
    augmented_data = np.roll(data, shift)
    return augmented_data

def change_pitch(data, sampling_rate, pitch_factor=2):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate)

def augment_data(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, sr=None)
        augmented_audios = [
            audio,
            add_noise(audio),
            shift(audio),
            change_pitch(audio, sample_rate, pitch_factor=2),
            stretch(audio, rate=0.8)
        ]
        mfccs_list = []
        for a in augmented_audios:
            mfccs = librosa.feature.mfcc(y=a, sr=sample_rate, n_mfcc=40)
            mfccs = np.mean(mfccs.T, axis=0)
            if mfccs.shape[0] == 40:  # Ensure all MFCCs have the same shape
                mfccs_list.append(mfccs)
        return mfccs_list
    except Exception as e:
        print(f"Error encountered while augmenting file: {file_path}, {e}")
        return []

# Verileri yükleme ve etiketleme fonksiyonu (Augment edilmiş verilerle)
def load_data(data_dir, save_augmented=False):
    features = []
    labels = []
    for label, emotion in enumerate(['calm', 'aggressive']):
        emotion_dir = os.path.join(data_dir, emotion)
        for file in os.listdir(emotion_dir):
            if file.endswith('.wav'):
                file_path = os.path.join(emotion_dir, file)
                augmented_features = augment_data(file_path)
                for feature in augmented_features:
                    features.append(feature)
                    labels.append(label)
    if save_augmented:
        with open('augmented_features.pkl', 'wb') as f:
            pickle.dump(features, f)
        with open('augmented_labels.pkl', 'wb') as f:
            pickle.dump(labels, f)
    return np.array(features), np.array(labels)

# Verileri yükleme
data_dir = 'xxxxxx' 
save_augmented = not os.path.exists('augmented_features.pkl')

if save_augmented:
    features, labels = load_data(data_dir, save_augmented=True)
else:
    with open('augmented_features.pkl', 'rb') as f:
        features = np.array(pickle.load(f))
    with open('augmented_labels.pkl', 'rb') as f:
        labels = np.array(pickle.load(f))

# Verilerin başarıyla yüklendiğini kontrol etme
print(f"Features shape: {features.shape}")
print(f"Labels shape: {labels.shape}")

# Verileri eğitim ve test setlerine ayırma
if features.size > 0 and labels.size > 0:
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Eğitim ve test setlerinin boyutlarını yazdırma
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # Etiketleri encode yapma
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)

    model = Sequential()
    model.add(Dense(256, input_shape=(40,), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    # Modeli derleme
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Modeli eğitme
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

    # Performans değerlendirmesi
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)
else:
    print("No data to train the model.")

# Eğitim sürecinin sonuçlarını görselleştirme
def plot_training_history(history):
    plt.figure(figsize=(12, 8))

    # Eğitim ve doğrulama kaybını çizdirme
    plt.subplot(2, 1, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Eğitim ve doğrulama doğruluğunu çizdirme
    plt.subplot(2, 1, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

# Eğitim sürecinin sonuçlarını görselleştirme
plot_training_history(history)

def plot_spectrogram(spectrogram, ax):
    # Spektrogramu çizdirme
    im = ax.imshow(spectrogram, aspect='auto', origin='lower', cmap='inferno')

    # Eksen etiketleri
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')
    ax.set_title('Spectrogram')

    # Renk çubuğu ekleme
    plt.colorbar(im, ax=ax)

def plot_confusion_matrix(y_true, y_pred, classes):
    # Karışıklık matrisini hesaplama
    cm = confusion_matrix(y_true, y_pred)

    # Karışıklık matrisini görselleştirme
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

# Modelinizin tahminleri ve gerçek etiketlerini kullanarak bir spektrogram görselleştirme
# Örnek olarak ilk test verisine odaklanalım
sample_index = 0
sample_features = X_test[sample_index]
sample_label = y_test[sample_index]

# Modelin tahmini
predicted_probabilities = model.predict(sample_features.reshape(1, -1))
predicted_label = np.argmax(predicted_probabilities)

# Spektrogramu çizdirme
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plot_spectrogram(sample_features.reshape((40, -1)), plt.gca())
plt.title(f'Sample Spectrogram (True Label: {sample_label}, Predicted Label: {predicted_label})')

# Karışıklık matrisini çizdirme
plt.subplot(2, 1, 2)
plot_confusion_matrix(y_test, np.argmax(model.predict(X_test), axis=1), classes=['calm', 'aggressive'])

# Mel-spektrogramu görselleştirme
def plot_mel_spectrogram(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    plt.figure(figsize=(10, 8))
    librosa.display.specshow(log_mel_spectrogram, sr=sample_rate, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.show()

# Örnek mel-spektrogramı görselleştirme
plot_mel_spectrogram('xxxxxx')  # Kendi ses dosyanızın yolunu buraya yazın
