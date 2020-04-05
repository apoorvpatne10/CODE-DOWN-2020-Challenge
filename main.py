import librosa
import numpy as np
import matplotlib.pyplot as plt
from librosa import display


file = "blues.00000.wav"

# Waveform
signal, sr = librosa.load(file, sr=22050)
# librosa.display.waveplot(signal, sr=sr)
# plt.xlabel("Time")
# plt.ylabel("Amplitude")
# plt.show()


# FFT -> Spectrum
fft = np.fft.fft(signal)
magnitude = np.abs(fft)
frequency = np.linspace(0, sr, len(magnitude))

left_freq = frequency[:int(len(frequency) / 2)]
left_magn = magnitude[:int(len(magnitude) / 2)]

# plt.plot(left_freq, left_magn)
# plt.xlabel('Frequency')
# plt.ylabel('Magnitude')
# plt.show()


# STFT - Spectrogram
n_fft = 2048
hop_length = 512

stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)
spectrogram = np.abs(stft)

log_spectrogram = librosa.amplitude_to_db(spectrogram)

# librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length)
# plt.xlabel("Time")
# plt.ylabel("Frequency")
# plt.colorbar()
# plt.show()


# MFCCs
MFCC = librosa.feature.mfcc(signal, n_fft=n_fft, hop_length=hop_length,
                            n_mfcc=13)
librosa.display.specshow(MFCC, sr=sr, hop_length=hop_length)
plt.title("MFCC")
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar()
plt.show()
