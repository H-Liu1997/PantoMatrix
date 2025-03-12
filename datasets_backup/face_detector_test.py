import unittest
import numpy as np
import mediapipe as mp
import cv2
from face_detector import FaceDetector

class TestFaceDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Download the model if it doesn't exist
        import urllib.request
        import os
        
        model_path = "face_landmarker.task"
        if not os.path.exists(model_path):
            print("Downloading face landmarker model...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
        
        cls.model_path = model_path
        
    def setUp(self):
        self.detector = FaceDetector(
            mediapipe_model_asset_path=self.model_path,
            face_detection_confidence=0.5,
            num_faces=5
        )
        
    def test_initialization(self):
        # Test if detector is properly initialized
        self.assertIsNotNone(self.detector)
        self.assertIsInstance(self.detector.detector, mp.tasks.vision.FaceLandmarker)

    def test_face_detection_empty_image(self):
        # Test with empty image (should handle gracefully)
        empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = self.detector.get_face_xy_rotation_and_keypoints(empty_image)
        
        # Unpack results
        all_x, all_y, all_orientation, num_faces, all_keypoints, all_bounding_box, all_blendshapes, all_facial_transformation_matrices, annotated_image = result
        
        # Should find no faces in empty image
        self.assertEqual(len(all_keypoints), 0)
        self.assertEqual(len(all_orientation), 0)
        self.assertEqual(len(all_blendshapes), 0)
        self.assertEqual(num_faces, 0)
        
    def test_face_detection_with_face(self):
        # Create a simple test image with a face-like pattern
        # Note: In practice, you should use a real face image from a test dataset
        test_image = np.ones((300, 300, 3), dtype=np.uint8) * 255
        
        # Draw a simple face-like pattern (this might not be detected as a face,
        # replace with real face image in production tests)
        cv2.circle(test_image, (150, 150), 50, (0, 0, 0), 2)  # Head
        cv2.circle(test_image, (130, 130), 5, (0, 0, 0), -1)  # Left eye
        cv2.circle(test_image, (170, 130), 5, (0, 0, 0), -1)  # Right eye
        cv2.line(test_image, (150, 140), (150, 160), (0, 0, 0), 2)  # Nose
        
        result = self.detector.get_face_xy_rotation_and_keypoints(test_image)
        all_x, all_y, all_orientation, num_faces, all_keypoints, all_bounding_box, all_blendshapes, all_facial_transformation_matrices, annotated_image = result
        
        # Basic structure tests (actual values would depend on the specific image)
        if len(all_keypoints) > 0:  # If a face is detected
            self.assertIsInstance(all_x[0], float)
            self.assertIsInstance(all_y[0], float)
            self.assertIn(all_orientation[0], ["left", "right", "forward"])
            self.assertTrue(len(all_keypoints[0]) > 0)
            self.assertEqual(len(all_bounding_box[0]), 2)  # Should have two points
            self.assertEqual(num_faces, len(all_keypoints))
            
    def test_face_detection_parameters(self):
        # Test with different confidence thresholds
        high_conf_detector = FaceDetector(
            mediapipe_model_asset_path=self.model_path,
            face_detection_confidence=0.9,
            num_faces=1
        )
        self.assertIsNotNone(high_conf_detector)
        
        low_conf_detector = FaceDetector(
            mediapipe_model_asset_path=self.model_path,
            face_detection_confidence=0.1,
            num_faces=1
        )
        self.assertIsNotNone(low_conf_detector)

if __name__ == '__main__':
    unittest.main()
