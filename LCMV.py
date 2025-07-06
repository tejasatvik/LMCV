import numpy as np
import librosa
import soundfile as sf
import os

# === Parameters ===
input_path = r"D:\Summer_Project\mixture_8ch.wav"
ref_mic = 0
n_fft = 1024
hop_length = 512
window = "hann"
spk1_duration_sec = 12.0
eps = 1e-10

# === Step 1: Load multichannel WAV ===
multi_sig, sr = librosa.load(input_path, sr=None, mono=False)
print(f"Loaded {multi_sig.shape[0]} channels at {sr} Hz")

# === Step 2: STFT ===
stft_all = librosa.stft(multi_sig, n_fft=n_fft, hop_length=hop_length, window=window)
n_channels, n_freq, n_frames = stft_all.shape

# === Step 3: RTF Estimation ===
spk1_end_frame = int(np.floor(spk1_duration_sec * sr / hop_length))
frame_energy = np.sum(np.abs(stft_all[ref_mic])**2, axis=0)
thr_spk1 = 0.1 * np.max(frame_energy[:spk1_end_frame])
thr_spk2 = 0.1 * np.max(frame_energy[spk1_end_frame:])
spk1_frames = np.where(frame_energy[:spk1_end_frame] >= thr_spk1)[0]
spk2_frames = np.where(frame_energy[spk1_end_frame:] >= thr_spk2)[0] + spk1_end_frame
X_spk1 = stft_all[:, :, spk1_frames]
X_spk2 = stft_all[:, :, spk2_frames]
X_ref_spk1 = stft_all[ref_mic, :, spk1_frames]
X_ref_spk2 = stft_all[ref_mic, :, spk2_frames]
numerator1 = np.sum(X_spk1 * np.conj(X_ref_spk1).transpose(1, 0)[None, :, :], axis=2)
denom1 = np.sum(np.abs(X_ref_spk1).T**2, axis=1) # shape (513,)
numerator2 = np.sum(X_spk2 * np.conj(X_ref_spk2).transpose(1, 0)[None, :, :], axis=2)
denom2 = np.sum(np.abs(X_ref_spk2).T**2, axis=1) # shape (513,)
rtf1 = numerator1 / (denom1[None, :] + eps)
rtf2 = numerator2 / (denom2[None, :] + eps)
rtf1[ref_mic, :] = 1.0
rtf2[ref_mic, :] = 1.0

# === Step 4: LCMV Beamforming ===
Y1 = np.zeros((n_freq, n_frames), dtype=complex)
Y2 = np.zeros((n_freq, n_frames), dtype=complex)
for f in range(n_freq):
    d1 = rtf1[:, f]
    d2 = rtf2[:, f]
    D = np.vstack([d1, d2]).T
    gram = D.conj().T @ D
    inv_gram = np.linalg.pinv(gram)
    w1 = D @ inv_gram[:, 0]
    w2 = D @ inv_gram[:, 1]
    Y1[f, :] = np.sum(np.conj(w1)[:, None] * stft_all[:, f, :], axis=0)
    Y2[f, :] = np.sum(np.conj(w2)[:, None] * stft_all[:, f, :], axis=0)

# === Step 5: Wiener Postfilter ===
P1 = np.abs(Y1)**2
P2 = np.abs(Y2)**2
G1 = P1 / (P1 + P2 + eps)
G2 = P2 / (P1 + P2 + eps)
Y1_post = Y1 * G1
Y2_post = Y2 * G2

# === Step 6: Inverse STFT ===
y1_est = librosa.istft(Y1_post, hop_length=hop_length, window=window, length=multi_sig.shape[1])
y2_est = librosa.istft(Y2_post, hop_length=hop_length, window=window, length=multi_sig.shape[1])

# === Save Outputs ===
out_dir = os.path.dirname(input_path)
sf.write(os.path.join(out_dir, "speaker3_estimate.wav"), y1_est, sr)
sf.write(os.path.join(out_dir, "speaker4_estimate.wav"), y2_est, sr)
print("Done. Output saved to:")
print(os.path.join(out_dir, "speaker3_estimate.wav"))
print(os.path.join(out_dir, "speaker4_estimate.wav"))
