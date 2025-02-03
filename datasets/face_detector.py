import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np


def convert_bbox_to_square_bbox(bbox, max_h, max_w, scale=1.0):
    # Calculate width, height, and max_size of the bounding box
    width = bbox[1][0] - bbox[0][0]
    height = bbox[1][1] - bbox[0][1]
    max_size = max(width, height) * scale

    # Calculate center of the bounding box
    center_x = (bbox[0][0] + bbox[1][0]) / 2
    center_y = (bbox[0][1] + bbox[1][1]) / 2

    # Calculate the left-up and right-bottom corners of the square bounding box
    half_size = max_size / 2
    left_top = [int(center_x - half_size), int(center_y - half_size)]
    right_bottom = [int(center_x + half_size), int(center_y + half_size)]

    # Ensure the square is within image bounds
    left_top[0] = max(0, left_top[0])  
    left_top[1] = max(0, left_top[1])
    right_bottom[0] = min(max_w, right_bottom[0])
    right_bottom[1] = min(max_h, right_bottom[1])

    # Return the new bounding box as a list of top-left and bottom-right coordinates
    return [left_top[0], left_top[1], right_bottom[0], right_bottom[1]]


def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in face_landmarks
            ]
        )

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style(),
        )
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style(),
        )

    return annotated_image


class FaceDetector:
    def __init__(self, mediapipe_model_asset_path, face_detection_confidence=0.5, num_faces=5):
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=mediapipe_model_asset_path,
                delegate=mp.tasks.BaseOptions.Delegate.GPU
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=num_faces,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            min_face_detection_confidence=face_detection_confidence,
            min_face_presence_confidence=face_detection_confidence,
            min_tracking_confidence=face_detection_confidence,
        )
        self.detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def get_face_xy_rotation_and_keypoints(self, image, mouth_bbox_scale = 1.2, eye_bbox_scale = 1.5, annotate_image: bool = False, save_vis=False):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    
        # get facial rotation
        results = self.detector.detect(mp_image)
        max_h, max_w = image.shape[:2]
        
        if annotate_image:
            annotated_image = draw_landmarks_on_image(image, results)
        else:
            annotated_image = None

        all_x = []
        all_y = []
        all_orientation = []
        all_keypoints = []
        all_bounding_box = []
        all_mouth_bounding_box = []
        all_eye_bounding_box = []
        all_face_contour = []
        all_blendshapes = []
        num_faces = len(results.face_landmarks)

        for face_blendshapes in results.face_blendshapes:
            blendshapes = [item.score for item in face_blendshapes]
            all_blendshapes.append(blendshapes)

        all_facial_transformation_matrices = results.facial_transformation_matrixes

        for face_landmarks in results.face_landmarks:
            keypoints = []
            bounding_box = []

            h, w = image.shape[0], image.shape[1]
            cx_min, cy_min = w, h
            cx_max, cy_max = 0, 0
            for idx, lm in enumerate(face_landmarks):
                # Clip landmarks if they go off the image
                cx, cy = int(np.clip(lm.x, 0, 1) * w), int(np.clip(lm.y, 0, 1) * h)

                if cx < cx_min:
                    cx_min = cx
                if cy < cy_min:
                    cy_min = cy
                if cx > cx_max:
                    cx_max = cx
                if cy > cy_max:
                    cy_max = cy

                keypoints.append((lm.x, lm.y, lm.z))

                if idx == 137:
                    right_cheek = (lm.x, lm.y, lm.z)
                if idx == 366:
                    left_cheek = (lm.x, lm.y, lm.z)
                if idx == 4:
                    nose = (lm.x, lm.y, lm.z)

            # get vector from middle of face to tip of nose
            face_middle = (
                (right_cheek[0] + left_cheek[0]) / 2.0,
                (right_cheek[1] + left_cheek[1]) / 2.0,
            )

            x = nose[0] - face_middle[0]
            y = nose[1] - face_middle[1]

            if x > 0.15:
                orientation = "left"
            elif x < -0.15:
                orientation = "right"
            else:
                orientation = "forward"

            bounding_box = [(cx_min, cy_min), (cx_max, cy_max)]

            all_keypoints.append(keypoints)
            all_bounding_box.append(bounding_box)
            all_x.append(x)
            all_y.append(y)
            all_orientation.append(orientation)

            # Get mouth bounding box (landmarks 13-17 and 308-312)
            mouth_landmarks = [
                61,
                146,
                146,
                91,
                91,
                181,
                181,
                84,
                84,
                17,
                17,
                314,
                314,
                405,
                405,
                321,
                321,
                375,
                375,
                291,
                61,
                185,
                185,
                40,
                40,
                39,
                39,
                37,
                37,
                0,
                0,
                267,
                267,
                269,
                269,
                270,
                270,
                409,
                409,
                291,
                78,
                95,
                95,
                88,
                88,
                178,
                178,
                87,
                87,
                14,
                14,
                317,
                317,
                402,
                402,
                318,
                318,
                324,
                324,
                308,
                78,
                191,
                191,
                80,
                80,
                81,
                81,
                82,
                82,
                13,
                13,
                312,
                312,
                311,
                311,
                310,
                310,
                415,
                415,
                308,
            ]
            # mouth_landmarks = [13, 14, 15, 16, 17, 308, 309, 310, 311, 312]
            mouth_x = [int(np.clip(face_landmarks[idx].x, 0, 1) * w) for idx in mouth_landmarks]
            mouth_y = [int(np.clip(face_landmarks[idx].y, 0, 1) * h) for idx in mouth_landmarks]
            mouth_bbox = [(min(mouth_x), min(mouth_y)), (max(mouth_x), max(mouth_y))]
            mouth_bbox = convert_bbox_to_square_bbox(mouth_bbox, max_h, max_w, scale=mouth_bbox_scale)

            # width = mouth_bbox[1][0] - mouth_bbox[0][0]
            # height = mouth_bbox[1][1] - mouth_bbox[0][1]
            # max_size = max(width, height) * 1.2
            # center_x = (mouth_bbox[0][0] + mouth_bbox[1][0]) / 2
            # center_y = (mouth_bbox[0][1] + mouth_bbox[1][1]) / 2
            # left_up = (int(center_x - max_size/2), int(center_y - max_size/2))
            # right_bottom = (int(center_x + max_size/2), int(center_y + max_size/2))
            # mouth_bbox = [left_up, right_bottom]

            all_mouth_bounding_box.append(mouth_bbox)

            # Get eye bounding boxes (left eye: landmarks 33-133, right eye: landmarks 362-263)
            left_eye_landmarks = [463, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 362]
            right_eye_landmarks = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
            
            left_eye_x = [int(np.clip(face_landmarks[idx].x, 0, 1) * w) for idx in left_eye_landmarks]
            left_eye_y = [int(np.clip(face_landmarks[idx].y, 0, 1) * h) for idx in left_eye_landmarks]
            left_eye_bbox = [(min(left_eye_x), min(left_eye_y)), (max(left_eye_x), max(left_eye_y))]
            left_eye_bbox = convert_bbox_to_square_bbox(left_eye_bbox, max_h, max_w, scale=eye_bbox_scale)
            
            right_eye_x = [int(np.clip(face_landmarks[idx].x, 0, 1) * w) for idx in right_eye_landmarks]
            right_eye_y = [int(np.clip(face_landmarks[idx].y, 0, 1) * h) for idx in right_eye_landmarks]
            right_eye_bbox = [(min(right_eye_x), min(right_eye_y)), (max(right_eye_x), max(right_eye_y))]
            right_eye_bbox = convert_bbox_to_square_bbox(right_eye_bbox, max_h, max_w, scale=eye_bbox_scale)

            eye_bbox = {"left_eye": left_eye_bbox, "right_eye": right_eye_bbox}
            
            all_eye_bounding_box.append(eye_bbox)
            
            face_contour = np.zeros_like(image)
            for landmark_id, landmark in enumerate(face_landmarks):
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                if cy >= max_h or cx >= max_w: continue
                if cy < 0 or cx < 0: continue
                face_contour[cy, cx] = (255, 255, 255)
            all_face_contour.append(face_contour)
            
            if save_vis:
                import cv2
                x_min, y_min, x_max, y_max = mouth_bbox
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
            
                for eye_key, bbox in eye_bbox.items():
                    x_min, y_min, x_max, y_max = bbox
                    color = (0, 0, 255)
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
                
                for landmark_id, landmark in enumerate(face_landmarks):
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    circle_size = 2
                    if landmark_id in mouth_landmarks:
                        cv2.circle(image, (cx, cy), circle_size, (0, 0, 255), -1)
                    elif landmark_id in left_eye_landmarks+right_eye_landmarks:
                        cv2.circle(image, (cx, cy), circle_size, (0, 255, 0), -1)
                    else:
                        cv2.circle(image, (cx, cy), circle_size, (255, 255, 255), -1)
                cv2.imwrite('image_detect.png', image[:,:,::-1])
                # import pdb; pdb.set_trace()

        return (
            all_x,
            all_y,
            all_orientation,
            num_faces,
            all_keypoints,
            all_bounding_box,
            all_mouth_bounding_box,
            all_eye_bounding_box,
            all_face_contour,
            all_blendshapes,
            all_facial_transformation_matrices,
            annotated_image,
        )
