from datasets.wav2vec_wrapper import wav2vec2_wrapper_new

def test_wav2vec2_wrapper_new():
    wa = wav2vec2_wrapper_new()
    wa.forward("configs/emo_inference/test_cases/test_cases_0501/seq0-wav0.wav", 15)

if __name__ == "__main__":
    test_wav2vec2_wrapper_new()