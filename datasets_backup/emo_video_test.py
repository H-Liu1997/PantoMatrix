import unittest
import torch
import numpy as np
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
from omegaconf import OmegaConf
import imageio
import soundfile as sf
from datasets.emo_video import CommonVideoDataset, EmoVideoDatasetAlex, EmoVideoDatasetAlexJoint
from PIL import Image
from torchvision import transforms


def create_dummy_video_data(root_path):
    """Helper function to create dummy video data for testing.
    
    Args:
        root_path (str): Root directory path to create test data
        
    Returns:
        tuple: (video_path, meta_path) paths as strings
        
    Raises:
        OSError: If unable to create directories or files
        ImportError: If required libraries are not available
    """
    try:
        # Create directories
        video_path = Path(root_path) / "videos1"
        meta_path = Path(root_path) / "meta1"
        video_path.mkdir(parents=True, exist_ok=True)
        meta_path.mkdir(parents=True, exist_ok=True)

        # Create dummy video file
        video_file = video_path / "test_video.mp4"
        if not video_file.exists():
            # Create a dummy video using numpy and save it
            frames = np.random.randint(0, 255, (30, 256, 256, 3), dtype=np.uint8)  # Smaller resolution for tests
            imageio.mimsave(str(video_file), frames, fps=30, quality=7)  # Lower quality for tests

        # Create a dummy audio file
        audio_file = video_path / "test_video.wav"
        if not audio_file.exists():
            # Create a dummy audio file with 1 second of silence
            sample_rate = 16000
            silence = np.zeros(sample_rate, dtype=np.float32)  # 1 second of silence
            sf.write(str(audio_file), silence, sample_rate)

        # Create dummy metadata with more realistic test data
        metadata = {
            "path_to_video": str(video_file),
            "tracks_synced": {
                0: {
                    "offset": 0.1,
                    "conf": 0.9,
                    "duration": 1.0
                }
            },
            "bounding_box_union": [[50, 200], [50, 200]],  # More realistic bbox
            "frame_count": 30,
            "fps": 30,
            "width": 256,
            "height": 256
        }
        
        meta_file = meta_path / "metadata.npz"
        if not meta_file.exists():
            np.savez(meta_file, metadata=np.array([metadata], dtype=object))

        return str(video_path), str(meta_path)
        
    except (OSError, ImportError) as e:
        raise RuntimeError(f"Failed to create dummy video data: {str(e)}")


# Mock classes for audio processing
class MockWav2Vec2Model:
    def __init__(self):
        self.feature_extractor = MagicMock()
        self.feature_extractor._freeze_parameters = MagicMock()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, input_value, output_hidden_states=True):
        # Return mock hidden states with expected shape
        batch_size = input_value.shape[0]
        sequence_length = 50  # Mock sequence length
        hidden_size = 768  # Standard wav2vec2 hidden size
        return {
            'hidden_states': [torch.randn(batch_size, sequence_length, hidden_size) 
                            for _ in range(13)]  # 13 layers including input embeddings
        }

class MockWav2Vec2FeatureExtractor:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def __call__(self, speech_array, sampling_rate=16000):
        if isinstance(speech_array, np.ndarray):
            # Convert numpy array to tensor with expected shape
            input_values = torch.from_numpy(speech_array).float()
        else:
            input_values = speech_array.float()
        
        if len(input_values.shape) == 1:
            input_values = input_values.unsqueeze(0)
            
        return {'input_values': input_values}

class MockAudioProcessor:
    def __init__(self, device='cpu'):
        self._processor = MockWav2Vec2FeatureExtractor()
        self._model = MockWav2Vec2Model()
        self.device = device

    def get_audio_features(self, audio_path):
        # Return mock features with expected shape
        return torch.randn(1, 50, 768)  # [batch_size, sequence_length, hidden_size]

    def get_audio_features_from_array(self, audio_array, sr=16000):
        # Return mock features with expected shape
        return torch.randn(1, 50, 768)  # [batch_size, sequence_length, hidden_size]

class Mockwav2vec2_wrapper_new:
    def __init__(self, device='cpu'):
        self._processor = MockWav2Vec2FeatureExtractor()
        self._model = MockWav2Vec2Model()
        self.device = device
        self._sampling_rate = 16000

    def get_audio_features(self, audio_path):
        # Return mock features with expected shape
        return torch.randn(1, 50, 768)  # [batch_size, sequence_length, hidden_size]

    def get_audio_features_from_array(self, audio_array, sr=16000):
        # Return mock features with expected shape
        return torch.randn(1, 50, 768)  # [batch_size, sequence_length, hidden_size]

class MockCLIPImageProcessor:
    def __init__(self):
        pass

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def preprocess(self, images, return_tensors="pt"):
        # Convert input to tensor if it's not already
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        elif isinstance(images, Image.Image):
            images = transforms.ToTensor()(images)
            
        # Ensure it's a batch
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
            
        return {'pixel_values': images}


@patch('transformers.CLIPImageProcessor.from_pretrained', MockCLIPImageProcessor.from_pretrained)
@patch('transformers.Wav2Vec2Model.from_pretrained', MockWav2Vec2Model.from_pretrained)
@patch('transformers.Wav2Vec2FeatureExtractor.from_pretrained', MockWav2Vec2FeatureExtractor.from_pretrained)
@patch('datasets.dataset_utils.get_audio_enc', return_value=MockAudioProcessor())
class TestCommonVideoDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are shared across all tests."""
        cls.temp_dir = tempfile.mkdtemp()
        video_path, meta_path = create_dummy_video_data(cls.temp_dir)
        
        config = {
            'data': {
                'meta_paths': [meta_path],
                'root_video_paths': [video_path],
                'audio_offset_thresh': 0.5,
                'audio_conf_thresh': 0.8,
                'mediapipe_mask': [False],
                'clip_after_augmentation': False,
                'zoom_out_ratio': [0.8, 1.2],
                'start_margin': 0,
                'end_margin': 0
            },
            'use_insightface_emb': False
        }
        cls.config = OmegaConf.create(config)
        cls.n_sample_frames = 8
        cls.width = 256
        cls.height = 256

    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures after all tests are done."""
        shutil.rmtree(cls.temp_dir)

    def test_initialization(self, mock_clip, mock_wav2vec_model, mock_wav2vec_feat, mock_get_audio_enc):
        """Test dataset initialization with basic parameters."""
        dataset = CommonVideoDataset(
            n_sample_frames=self.n_sample_frames,
            width=self.width,
            height=self.height,
            cfg=self.config,
            audio_fea_type='wav2vec2_type2'
        )
        self.assertEqual(dataset.n_sample_frames, self.n_sample_frames)
        self.assertEqual(dataset.width, self.width)
        self.assertEqual(dataset.height, self.height)
        self.assertIsNotNone(dataset.pixel_transform)
        self.assertIsNotNone(dataset.cond_transform)

    def test_transforms(self, mock_clip, mock_wav2vec_model, mock_wav2vec_feat, mock_get_audio_enc):
        """Test that transforms are properly configured."""
        dataset = CommonVideoDataset(
            n_sample_frames=self.n_sample_frames,
            width=self.width,
            height=self.height,
            cfg=self.config,
            audio_fea_type='wav2vec2_type2'
        )
        # Create a dummy image tensor
        dummy_image = torch.randn(3, self.height, self.width)
        
        # Test pixel transform
        transformed = dataset.pixel_transform(dummy_image)
        self.assertEqual(transformed.shape, (3, self.height, self.width))
        self.assertTrue(torch.all(transformed >= -1) and torch.all(transformed <= 1))

        # Test conditional transform
        transformed = dataset.cond_transform(dummy_image)
        self.assertEqual(transformed.shape, (3, self.height, self.width))

    def test_getitem(self, mock_clip, mock_wav2vec_model, mock_wav2vec_feat, mock_get_audio_enc):
        """Test the __getitem__ functionality."""
        dataset = CommonVideoDataset(
            n_sample_frames=self.n_sample_frames,
            width=self.width,
            height=self.height,
            cfg=self.config,
            audio_fea_type='wav2vec2_type2'
        )

        # Test getting an item from the dataset
        item = dataset[0]
        
        # Check if all expected keys are present
        expected_keys = ['video', 'audio_feature', 'frame_indices']
        for key in expected_keys:
            self.assertIn(key, item)
        
        # Check video tensor shape
        self.assertEqual(item['video'].shape, (self.n_sample_frames, 3, self.height, self.width))
        
        # Check audio feature shape
        self.assertEqual(len(item['audio_feature'].shape), 2)  # Should be 2D tensor
        
        # Check frame indices
        self.assertEqual(len(item['frame_indices']), self.n_sample_frames)


@patch('transformers.CLIPImageProcessor.from_pretrained', MockCLIPImageProcessor.from_pretrained)
@patch('transformers.Wav2Vec2Model.from_pretrained', MockWav2Vec2Model.from_pretrained)
@patch('transformers.Wav2Vec2FeatureExtractor.from_pretrained', MockWav2Vec2FeatureExtractor.from_pretrained)
@patch('datasets.dataset_utils.get_audio_enc', return_value=MockAudioProcessor())
class TestEmoVideoDatasetAlex(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are shared across all tests."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.video_path, cls.meta_path = create_dummy_video_data(cls.temp_dir)
        
        config = {
            'data': {
                'meta_paths': [cls.meta_path],
                'root_video_paths': [cls.video_path],
                'audio_offset_thresh': 0.5,
                'audio_conf_thresh': 0.8,
                'mediapipe_mask': [False],
                'clip_after_augmentation': False,
                'zoom_out_ratio': [0.8, 1.2],
                'start_margin': 0,
                'end_margin': 0
            },
            'use_insightface_emb': False
        }
        cls.config = OmegaConf.create(config)
        cls.n_sample_frames = 8
        cls.width = 256
        cls.height = 256

    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures after all tests are done."""
        shutil.rmtree(cls.temp_dir)

    def test_initialization(self, mock_clip, mock_wav2vec_model, mock_wav2vec_feat, mock_get_audio_enc):
        """Test dataset initialization."""
        dataset = EmoVideoDatasetAlex(
            n_sample_frames=self.n_sample_frames,
            width=self.width,
            height=self.height,
            cfg=self.config,
            audio_fea_type='wav2vec2_type2'
        )
        
        self.assertEqual(dataset.n_sample_frames, self.n_sample_frames)
        self.assertEqual(dataset.width, self.width)
        self.assertEqual(dataset.height, self.height)
        self.assertIsNotNone(dataset.pixel_transform)
        self.assertIsNotNone(dataset.cond_transform)

    def test_check_bbox(self, mock_clip, mock_wav2vec_model, mock_wav2vec_feat, mock_get_audio_enc):
        """Test bbox validation function."""
        dataset = EmoVideoDatasetAlex(
            n_sample_frames=self.n_sample_frames,
            width=self.width,
            height=self.height,
            cfg=self.config,
            audio_fea_type='wav2vec2_type2'
        )

        # Test valid bbox
        valid_bbox = [[0, 100], [0, 100]]
        self.assertTrue(dataset._check_bbox(valid_bbox))

        # Test invalid bbox - empty
        invalid_bbox = []
        self.assertFalse(dataset._check_bbox(invalid_bbox))

        # Test invalid bbox - wrong format
        invalid_bbox = [0, 100, 0, 100]
        self.assertFalse(dataset._check_bbox(invalid_bbox))

        # Test invalid bbox - negative values
        invalid_bbox = [[-10, 100], [0, 100]]
        self.assertFalse(dataset._check_bbox(invalid_bbox))

    def test_getitem(self, mock_clip, mock_wav2vec_model, mock_wav2vec_feat, mock_get_audio_enc):
        """Test the __getitem__ functionality."""
        dataset = EmoVideoDatasetAlex(
            n_sample_frames=self.n_sample_frames,
            width=self.width,
            height=self.height,
            cfg=self.config,
            audio_fea_type='wav2vec2_type2'
        )

        item = dataset[0]
        
        # Check if all expected keys are present
        expected_keys = ['video', 'audio_feature', 'frame_indices']
        for key in expected_keys:
            self.assertIn(key, item)
        
        # Check video tensor shape
        self.assertEqual(item['video'].shape, (self.n_sample_frames, 3, self.height, self.width))
        
        # Check audio feature shape
        self.assertEqual(len(item['audio_feature'].shape), 2)  # Should be 2D tensor
        
        # Check frame indices
        self.assertEqual(len(item['frame_indices']), self.n_sample_frames)


@patch('transformers.CLIPImageProcessor.from_pretrained', MockCLIPImageProcessor.from_pretrained)
@patch('transformers.Wav2Vec2Model.from_pretrained', MockWav2Vec2Model.from_pretrained)
@patch('transformers.Wav2Vec2FeatureExtractor.from_pretrained', MockWav2Vec2FeatureExtractor.from_pretrained)
@patch('datasets.dataset_utils.get_audio_enc', return_value=MockAudioProcessor())
class TestEmoVideoDatasetAlexJoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are shared across all tests."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.video_path, cls.meta_path = create_dummy_video_data(cls.temp_dir)
        
        config = {
            'data': {
                'meta_paths': [cls.meta_path],
                'root_video_paths': [cls.video_path],
                'audio_offset_thresh': 0.5,
                'audio_conf_thresh': 0.8,
                'mediapipe_mask': [False],
                'clip_after_augmentation': False,
                'zoom_out_ratio': [0.8, 1.2],
                'start_margin': 0,
                'end_margin': 0
            },
            'use_insightface_emb': False
        }
        cls.config = OmegaConf.create(config)
        cls.n_sample_frames = 8
        cls.width = 256
        cls.height = 256

    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures after all tests are done."""
        shutil.rmtree(cls.temp_dir)

    def test_initialization(self, mock_clip, mock_wav2vec_model, mock_wav2vec_feat, mock_get_audio_enc):
        """Test dataset initialization."""
        dataset = EmoVideoDatasetAlexJoint(
            n_sample_frames=self.n_sample_frames,
            width=self.width,
            height=self.height,
            cfg=self.config,
            audio_fea_type='wav2vec2_type2'
        )
        
        self.assertEqual(dataset.n_sample_frames, self.n_sample_frames)
        self.assertEqual(dataset.width, self.width)
        self.assertEqual(dataset.height, self.height)
        self.assertIsNotNone(dataset.pixel_transform)
        self.assertIsNotNone(dataset.cond_transform)

    def test_getitem(self, mock_clip, mock_wav2vec_model, mock_wav2vec_feat, mock_get_audio_enc):
        """Test the __getitem__ functionality."""
        dataset = EmoVideoDatasetAlexJoint(
            n_sample_frames=self.n_sample_frames,
            width=self.width,
            height=self.height,
            cfg=self.config,
            audio_fea_type='wav2vec2_type2'
        )

        item = dataset[0]
        
        # Check if all expected keys are present
        expected_keys = ['video', 'audio_feature', 'frame_indices']
        for key in expected_keys:
            self.assertIn(key, item)
        
        # Check video tensor shape
        self.assertEqual(item['video'].shape, (self.n_sample_frames, 3, self.height, self.width))
        
        # Check audio feature shape
        self.assertEqual(len(item['audio_feature'].shape), 2)  # Should be 2D tensor
        
        # Check frame indices
        self.assertEqual(len(item['frame_indices']), self.n_sample_frames)


if __name__ == '__main__':
    unittest.main()
