from .motion_gen import INFPDiffusionTransformerBlock,INFPDiffusionTransformer
from .motion_guider import MotionGuider
from .audio_processor import AudioProcessor
import torch
import time
import unittest
import os
import time
import scipy
import numpy as np

class TestMotionGuider(unittest.TestCase):
    def test_forward_pass(self):
        BATCH_SIZE = 2
        NUM_FRAMES_AUDIO = 15
        AUDIO_FEATURE_DIM = 768
        REF_LATENTS_FRAMES = 15
        LATENT_HEIGHT = 32
        LATENT_WIDTH = 32
        LATENT_DIM = 32
        ref_latents = torch.randn((BATCH_SIZE*REF_LATENTS_FRAMES,LATENT_DIM,LATENT_HEIGHT,LATENT_WIDTH))
        audio_self_feature = torch.randn((BATCH_SIZE,NUM_FRAMES_AUDIO,AUDIO_FEATURE_DIM))
        audio_other_feature = torch.randn((BATCH_SIZE,NUM_FRAMES_AUDIO,AUDIO_FEATURE_DIM))  
        motion_guider = MotionGuider(query_dim=AUDIO_FEATURE_DIM,enable_style_modulation=True,latents_dim=LATENT_DIM)
        motion_features = motion_guider(audio_self_feature,audio_other_feature,ref_latents,REF_LATENTS_FRAMES)
        self.assertEqual(motion_features.shape,(BATCH_SIZE,NUM_FRAMES_AUDIO,AUDIO_FEATURE_DIM))

class TestAudioEncoder(unittest.TestCase):
    def test_forward_pass(self):
        DURATION = 3 # audio for 3 seconds
        SAMPLE_RATE = 16000
        USE_LAST_LATENTS = True
        FPS = 15
        def save_empty_audio(duration, sample_rate, output_path):
            empty_audio = np.zeros(int(duration * sample_rate)).astype(np.int16)
            scipy.io.wavfile.write(output_path, sample_rate, empty_audio)
            return output_path
        audio_processor = AudioProcessor(
            SAMPLE_RATE,
            FPS,
            "facebook/wav2vec2-base-960h",
            USE_LAST_LATENTS,
        )
        save_empty_audio(DURATION, SAMPLE_RATE, "./test.wav")
        audio_emb = audio_processor.get_embedding("test.wav") # first forward for build graph
        # now start to benchmark the audio_processor
        s = time.perf_counter()
        audio_emb = audio_processor.get_embedding("test.wav")
        print(f"{audio_emb.shape=}")
        print(f"embedded for {DURATION}s audio, {SAMPLE_RATE} sample rate, cost: {time.perf_counter()-s}s")
        if os.path.exists('./test.wav'):
            os.remove('./test.wav')
        self.assertEqual(audio_emb.shape[0],DURATION * FPS)

class TestDiffusionTransformer(unittest.TestCase):
    def test_forward_pass(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        BATCH_SIZE = 2 # number of batches
        FRAME_SIZE = 5 # number of frames in a batch
        N_AUDIO_FRAMES = 10 # number of frames in condition
        LATENT_HEIGHT = 32
        LATENT_WIDTH = 32
        LATENT_DIM = 32
        AUDIO_DIM = 768
        N_PAST_FRMAES = 2 # number of past frames
        dummy_hidden_latents = torch.randn((BATCH_SIZE*FRAME_SIZE,LATENT_DIM,LATENT_HEIGHT,LATENT_WIDTH)).to(device)
        dummy_past_latents = torch.randn((BATCH_SIZE*N_PAST_FRMAES,LATENT_DIM,LATENT_HEIGHT,LATENT_WIDTH)).to(device)
        dummy_audio_self = torch.randn((BATCH_SIZE,N_AUDIO_FRAMES,AUDIO_DIM)).to(device)
        dummy_audio_other = torch.randn((BATCH_SIZE,N_AUDIO_FRAMES,AUDIO_DIM)).to(device)
        denoise_transformer = INFPDiffusionTransformer(latents_dim=LATENT_DIM,audio_dim=AUDIO_DIM).to(device)
        dummy_timesteps = torch.randint(
            0,
            50, # training timestep
            (BATCH_SIZE,),
            device=device,
        )
        pred = denoise_transformer(
            dummy_hidden_latents,
            dummy_audio_self,
            dummy_audio_other,
            dummy_past_latents,
            FRAME_SIZE,
            N_PAST_FRMAES,
            timestep=dummy_timesteps
        )
        self.assertEqual(pred.shape,dummy_hidden_latents.shape)


if __name__ == '__main__':
    TestDiffusionTransformer().test_forward_pass()
    # TestAudioEncoder().test_forward_pass()
    # TestMotionGuider().test_forward_pass()
    # unittest.main()