# cache motion latent from pretrained checkpoints
import os
import numpy as np
import librosa
import soundfile as sf

cache_path = "./BEAT2/cache_latent/"
audio_folder = "./BEAT2/cache_audio/"
os.makedirs(cache_path, exist_ok=True)
os.makedirs(audio_folder, exist_ok=True)

audio_path = "/home/weili/haiyang/PantoMatrix/BEAT2/beat_english_v2.0.0/wave16k/1_wayne_0_1_1.wav"
audio, sr = librosa.load(audio_path, sr=16000)

for i in range(128):
    n = np.random.randint(200, 401)
    duration = n / 30
    audio_length = int(duration * sr)

    if audio_length > audio.shape[0]:
        print(f"Audio too short for test_random_{i}, truncating duration.")
        audio_length = audio.shape[0]

    fake_audio = audio[:audio_length]
    fake_audio_path = os.path.join(audio_folder, f"test_random_{i}.wav")
    sf.write(fake_audio_path, fake_audio, sr)

    random_data = np.random.rand(audio_length // sr * 30, 256)
    filename = f"test_random_{i}.npz"
    np.savez(os.path.join(cache_path, filename), random_data=random_data)

