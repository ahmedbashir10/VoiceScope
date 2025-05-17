import numpy as np
import scipy.signal
import scipy.fftpack
import matplotlib.pyplot as plt
import librosa
import librosa.filters
from scipy.fftpack import dct


def enframe(samples, samplerate, win_len_ms=20, win_shift_ms=10):
    frame_len = int(win_len_ms * samplerate / 1000)
    frame_shift = int(win_shift_ms * samplerate / 1000)
    num_frames = (len(samples) - frame_len) // frame_shift + 1
    frames = np.stack([samples[i*frame_shift:i*frame_shift+frame_len] for i in range(num_frames)])
    return frames


def pre_emphasis(frames, coeff=0.97):
    return scipy.signal.lfilter([1, -coeff], [1], frames, axis=1)


def apply_hamming(frames):
    window = scipy.signal.hamming(frames.shape[1], sym=False)
    return frames * window

def power_spectrum(frames, n_fft=512):
    fft_result = np.fft.rfft(frames, n=n_fft)
    power_spec = np.abs(fft_result) ** 2
    return power_spec



def log_mel_spectrum(power_spec, samplerate, n_mels=26, n_fft=512):
    mel_filter = librosa.filters.mel(sr=samplerate, n_fft=n_fft, n_mels=n_mels)
    mel_energy = np.dot(power_spec, mel_filter.T)
    mel_energy = np.where(mel_energy == 0, np.finfo(float).eps, mel_energy)  # avoid log(0)
    return np.log(mel_energy)


def compute_mfcc(log_mel_energy, num_ceps=13):
    return dct(log_mel_energy, type=2, axis=1, norm='ortho')[:, :num_ceps]


def extract_mfcc(samples, sr):
    frames = enframe(samples, sr)
    emphasized = pre_emphasis(frames)
    windowed = apply_hamming(emphasized)
    power_spec = power_spectrum(windowed)
    log_mel = log_mel_spectrum(power_spec, sr)
    mfccs = compute_mfcc(log_mel)
    return mfccs
