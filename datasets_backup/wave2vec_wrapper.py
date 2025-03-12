import json
import math
import os
import random
import tempfile
from typing import List

import librosa
import moviepy.editor as mp
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchvision.transforms as transforms
from decord import AudioReader, VideoReader, cpu, gpu
from scipy import interpolate
from transformers import (Wav2Vec2Config, Wav2Vec2FeatureExtractor,
                          Wav2Vec2Processor)

if os.path.exists("./pretrained_weights"):
    PRETRAINED_WEIGHT_ROOT = "./pretrained_weights"
else:
    PRETRAINED_WEIGHT_ROOT = "/mnt/weka/pretrained_weights"


def tonp(x):
    return x.detach().cpu().numpy()

def fix_ckpt_error(in_path, out_path):
    from omegaconf import DictConfig, OmegaConf, open_dict

    cp = torch.load(in_path)
    cfg = DictConfig(cp["cfg"])

    if False:
        for k, v in OmegaConf.to_container(cfg, resolve=True).items():
            if not isinstance(v, dict):
                continue
            for key, _ in v.items():
                if key == "eval_wer":
                    print(k)
                    break

    wrong_key = ["eval_wer", "eval_wer_config", "eval_wer_tokenizer", "eval_wer_post_process", "autoregressive"]
    with open_dict(cfg):
        for k in wrong_key:
            cfg.task.pop(k)
    cp["cfg"] = cfg
    torch.save(cp, out_path)


# fix_ckpt_error('/mnt/weka/pretrained_weights/xlsr_53_56k.pt', '/mnt/weka/pretrained_weights/xlsr_53_56k_new.pt')


def get_wav2vec_model():
    audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    wav2vec_model.eval()
    return wav2vec_model, audio_processor


def get_wav2vec_model2():
    # https://huggingface.co/spaces/sriramelango/Social_Classification_Public/blob/main/fairseq/examples/wav2vec/README.md
    cp_path = f"{PRETRAINED_WEIGHT_ROOT}/xlsr_53_56k_new.pt"
    import fairseq

    # from fairseq.models.wav2vec import Wav2Vec2Model
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
    model = model[0]
    model.eval()
    return model

def get_wav2vec_model3():
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    print("Sample Rate:", bundle.sample_rate)
    print("Labels:", bundle.get_labels())
    model = bundle.get_model().cuda(0)
    return model

def load_audio_from_mp4(mp4_path):
    video = mp.VideoFileClip(mp4_path)
    audio = video.audio
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_file.close()
    audio.write_audiofile(
        temp_file.name,
    )
    y, sr = librosa.load(temp_file.name, sr=16000)
    os.unlink(temp_file.name)
    return y

def load_audio_from_mp4_2(video_path):
    import pydub

    audio_seg = pydub.AudioSegment.from_file(video_path, "mp4", frame_rate=16000)
    audio_segment_resampled = audio_seg.set_frame_rate(16000)
    audio_segment_resampled.export("audio_output_16k.wav", format="wav")

    ar = AudioReader(video_path, ctx=cpu(0), mono=False)
    wav_input_16khz = ar[0]
    return None

def resample(tensor, old_rate, new_rate):
    new_length = int(tensor.shape[0] * (new_rate / old_rate))
    x_old = np.linspace(0, tensor.shape[0], tensor.shape[0])
    x_new = np.linspace(0, tensor.shape[0], new_length)
    resampled_tensor = np.empty((new_length, tensor.shape[1]))
    for i in range(tensor.shape[1]):
        interpolator = interpolate.interp1d(x_old, tensor[:, i])
        resampled_tensor[:, i] = interpolator(x_new)
    return resampled_tensor

def get_sliced_feature(feature_array, vid_idx, audio_feat_length=[2, 2], fps=25):
    """
    Get sliced features based on a given index
    :param feature_array:
    :param start_idx: the start index of the feature
    :param audio_feat_length:
    :return:
    """
    length = len(feature_array)
    selected_feature = []
    selected_idx = []

    center_idx = int(vid_idx * 50 / fps)
    left_idx = center_idx - audio_feat_length[0] * 2
    right_idx = center_idx + (audio_feat_length[1] + 1) * 2

    for idx in range(left_idx, right_idx):
        idx = max(0, idx)
        idx = min(length - 1, idx)
        x = feature_array[idx]
        selected_feature.append(x)
        selected_idx.append(idx)

    selected_feature = torch.cat(selected_feature, axis=0)
    return selected_feature, selected_idx

def feature2chunks(feature_array, fps, audio_feat_length=[2, 2]):
    whisper_chunks = []
    whisper_idx_multiplier = 50.0 / fps
    i = 0
    while 1:
        start_idx = int(i * whisper_idx_multiplier)
        if start_idx >= len(feature_array):
            break
        selected_feature, selected_idx = get_sliced_feature(
            feature_array=feature_array, vid_idx=i, audio_feat_length=audio_feat_length, fps=fps
        )
        # print(f"i:{i},selected_idx {selected_idx}")
        whisper_chunks.append(selected_feature)
        i += 1
    whisper_chunks = torch.stack(whisper_chunks, dim=0)
    return whisper_chunks

class wav2vec2_wrapper_new:
    def __init__(
        self, device="cuda", sampling_rate=16000, model_path=f"{PRETRAINED_WEIGHT_ROOT}/wav2vec2-base-960h"
    ) -> None:
        self.device = device

        from transformers import Wav2Vec2Model

        audio_encoder = Wav2Vec2Model.from_pretrained(model_path, local_files_only=True)
        audio_encoder.feature_extractor._freeze_parameters()
        self.audio_encoder = audio_encoder.to(device)
        self.audio_encoder.eval()

        self._processor = Wav2Vec2FeatureExtractor.from_pretrained(model_path, local_files_only=True)
        self._sampling_rate = sampling_rate

    def forward(
        self,
        wav_file,
        fps,
        cvt_to_chunk=True,
        only_last_features=False,
    ):
        speech_array, sampling_rate = librosa.load(wav_file, sr=self._sampling_rate)
        input_value = np.squeeze(self._processor(speech_array, sampling_rate=sampling_rate).input_values)

        input_value = torch.from_numpy(input_value).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            embeddings = self.audio_encoder(input_value, output_hidden_states=True)

        # import ipdb;ipdb.set_trace()

        if only_last_features:
            fea = embeddings.last_hidden_state[0].unsqueeze(1)
        else:
            fea = embeddings.hidden_states
            fea = torch.stack(fea, dim=2)[0]  # T, 13, 768
        if cvt_to_chunk:
            fea = feature2chunks(fea, fps, audio_feat_length=[2, 2])
        return fea

class wav2vec2_wrapper:
    def __init__(
        self, device="cuda", sampling_rate=16000, model_path=f"{PRETRAINED_WEIGHT_ROOT}/wav2vec2-base-960h"
    ) -> None:
        self.device = device
        from src.dataset.wav2vec2_mod import Wav2Vec2ModelMOD

        audio_encoder = Wav2Vec2ModelMOD.from_pretrained(model_path, local_files_only=True)
        audio_encoder.feature_extractor._freeze_parameters()
        self.audio_encoder = audio_encoder.to(device)
        self.audio_encoder.eval()

        # audio_encoder_config = Wav2Vec2Config.from_pretrained(model_path, local_files_only=True)
        # hidden_size = audio_encoder_config.hidden_size # 768
        # self.data_preprocessor = DataProcessor(sampling_rate, wav2vec_model_path)
        self._processor = Wav2Vec2FeatureExtractor.from_pretrained(model_path, local_files_only=True)
        self._sampling_rate = sampling_rate

    def forward(
        self,
        wav_file,
        fps,
        only_last_features=True,
    ):
        audio_len = None
        # import ipdb;ipdb.set_trace()
        # input_value = self.data_preprocessor.extract_feature(wav_file)
        speech_array, sampling_rate = librosa.load(wav_file, sr=self._sampling_rate)
        input_value = np.squeeze(self._processor(speech_array, sampling_rate=sampling_rate).input_values)
        seq_len = math.ceil(len(input_value) / self._sampling_rate * fps)  # T*fps

        input_value = torch.from_numpy(input_value).float().unsqueeze(0).to(self.device)

        from src.dataset.wav2vec2_mod import get_mask_from_lengths

        attention_mask = ~get_mask_from_lengths(audio_len) if audio_len else None

        with torch.no_grad():
            embeddings = self.audio_encoder(
                input_value, seq_len=seq_len, output_hidden_states=True, attention_mask=attention_mask
            )

        # import ipdb;ipdb.set_trace()
        if only_last_features:
            hidden_states = embeddings.last_hidden_state  # [1, T*fps, 768]
        else:
            hidden_states = sum(embeddings.hidden_states) / len(embeddings.hidden_states)
            # embeddings.hidden_states: len 13
            # embeddings.hidden_states[0] ([1, T*fps, 768])

        wav_fea = hidden_states.detach()[0]

        return wav_fea


def test_wav2vec_speech_to_text():
    import os

    import librosa
    import moviepy.editor as mp
    import torch
    from IPython.display import Audio
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

    tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

    model_path = f"{PRETRAINED_WEIGHT_ROOT}/wav2vec2-base-960h"
    model = Wav2Vec2ForCTC.from_pretrained(model_path)
    tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_path, local_files_only=True)

    path = "configs/emo_inference/test_cases/compare_emo/emo_offical_sora_5s.wav"
    # Load the audio with the librosa library
    input_audio, _ = librosa.load(path, sr=16000)

    # Tokenize the audio
    input_values = tokenizer(input_audio, return_tensors="pt", padding="longest").input_values

    # Feed it through Wav2Vec & choose the most probable tokens
    with torch.no_grad():
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)

    # Decode & add to our caption string
    transcription = tokenizer.batch_decode(predicted_ids)[0]
    print(f"{transcription=}")