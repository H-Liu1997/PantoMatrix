import os 
import random
import sys
import numpy as np
import imageio
import torch
import torchvision.transforms.v2 as transforms
from decord import VideoReader, cpu, gpu
from omegaconf import OmegaConf
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import ffmpeg
import json 
import librosa
import cv2
import torch.nn.functional as F

from datasets.emo_image import (get_mask, get_move_area, get_scale_bbox,
                        mediapipe2s3fd, random_crop_with_bbox, scale_bbox,
                        tonp)
from datasets.face_detector import FaceDetector, convert_bbox_to_square_bbox
from datasets.dataset_utils import generate_crop_bounding_box, crop_from_bbox

def get_mask(bbox, hd, wd, scale=1.0, return_pil=True):
    if min(bbox) < 0:
        raise Exception("Invalid mask")
    # sontime bbox is like this: array([ -8.84635544, 216.97692871, 192.20074463, 502.83700562])
    bbox = scale_bbox(bbox, hd, wd, scale=scale)
    bbox_x0, bbox_y0, bbox_x1, bbox_y1 = [int(ii) for ii in bbox]
    # tgt_pose = np.zeros_like(tgt_img.asnumpy())
    tgt_pose = np.zeros((hd, wd, 3))
    tgt_pose[bbox_y0:bbox_y1, bbox_x0:bbox_x1, :] = 255.0
    if return_pil:
        tgt_pose_pil = Image.fromarray(tgt_pose.astype(np.uint8))
        return tgt_pose_pil
    return tgt_pose

def get_face_box(landmark_name, face_landmarks, max_w, max_h, eye_bbox_scale=1.5):
    # print(face_landmarks)
    if landmark_name == 'left_eye':
        landmarks = [463, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 362]
    elif landmark_name == 'right_eye':
        landmarks = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
    elif landmark_name == 'mouth':
        landmarks = [61, 146, 146, 91, 91, 181, 181, 84, 84, 17, 17, 314, 314, 405, 405, 321, 321, 375, 375, 291, 61, 185, 185, 40, 40, 39, 39, 37, 37, 0, 0, 267, 267, 269, 269, 270, 270, 409, 409, 291, 78, 95, 95, 88, 88, 178, 178, 87, 87, 14, 14, 317, 317, 402, 402, 318, 318, 324, 324, 308, 78, 191, 191, 80, 80, 81, 81, 82, 82, 13, 13, 312, 312, 311, 311, 310, 310, 415, 415, 308]
    
    landmarks_x = [int(np.clip(face_landmarks[idx][0], 0, max_w)) for idx in landmarks]
    landmarks_y = [int(np.clip(face_landmarks[idx][1], 0, max_h)) for idx in landmarks]
    # print(landmarks_x, landmarks_y)
    bbox = [(min(landmarks_x), min(landmarks_y)), (max(landmarks_x), max(landmarks_y))]
    # print(bbox)
    bbox = convert_bbox_to_square_bbox(bbox, max_h, max_w, scale=eye_bbox_scale)
    return bbox

def get_face_contour(face_landmarks, h, w):
    face_contour = np.zeros((h, w, 3), dtype=np.uint8)
    face_contour_bbox = []
    for landmark_id, landmark in enumerate(face_landmarks):
        cx, cy, _ = landmark
        cx, cy  = int(cx), int(cy)
        if cy >= h or cx >= w: continue
        if cy < 0 or cx < 0: continue
        face_contour[cy, cx] = (255, 255, 255)
    return face_contour

def check_bbox(bbox):
    if not np.isfinite(bbox).all():
        raise Exception(f"Some of the bboxes have non-finite values.")
    bbox_extend = bbox[:, 1] - bbox[:, 0]
    if bbox_extend.min() <= 0:
        raise Exception(f"0, Some of the bboxes are invalid -- they extend zero area.")
     
def scale_bbox(bbox, scale=1.0):
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = (x1 - x0) * scale, (y1 - y0) * scale
    return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]

def convert_bbox_to_square_bbox(bbox, max_h, max_w, scale=1.0):
    if isinstance(bbox, (list, tuple)) and len(bbox) == 2:
        (x0, y0), (x1, y1) = bbox
    else:
        x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = x1 - x0, y1 - y0
    side = max(w, h) * scale
    new_x0 = max(0, cx - side/2)
    new_y0 = max(0, cy - side/2)
    new_x1 = min(max_w, cx + side/2)
    new_y1 = min(max_h, cy + side/2)
    return [int(new_x0), int(new_y0), int(new_x1), int(new_y1)]

def vectorized_get_mask(bboxes, hd, wd):
    # bboxes: (N, 4)
    N = bboxes.shape[0]
    Y = np.arange(hd).reshape(1, hd, 1)
    X = np.arange(wd).reshape(1, 1, wd)
    x0 = bboxes[:, 0].reshape(N, 1, 1)
    y0 = bboxes[:, 1].reshape(N, 1, 1)
    x1 = bboxes[:, 2].reshape(N, 1, 1)
    y1 = bboxes[:, 3].reshape(N, 1, 1)
    mask = (((X >= x0) & (X < x1) & (Y >= y0) & (Y < y1)).astype(np.uint8)) * 255
    return np.repeat(mask[:, :, :, np.newaxis], 3, axis=3)

def vectorized_get_face_contour(face_keypoints, hd, wd):
    N, num_landmarks, _ = face_keypoints.shape
    contour = np.zeros((N, hd, wd), dtype=np.uint8)
    x_coords = np.clip(face_keypoints[:, :, 0].astype(np.int32), 0, wd - 1)
    y_coords = np.clip(face_keypoints[:, :, 1].astype(np.int32), 0, hd - 1)
    idx = np.arange(N)[:, None]
    contour[idx, y_coords, x_coords] = 255
    return np.repeat(contour[:, :, :, np.newaxis], 3, axis=3)

def vectorized_get_face_box(face_keypoints, indices, max_w, max_h, scale):
    pts = face_keypoints[:, indices, :2]
    x_min = np.min(pts[:, :, 0], axis=1)
    y_min = np.min(pts[:, :, 1], axis=1)
    x_max = np.max(pts[:, :, 0], axis=1)
    y_max = np.max(pts[:, :, 1], axis=1)
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    side = np.maximum(x_max - x_min, y_max - y_min) * scale
    new_x0 = np.clip(cx - side/2, 0, max_w)
    new_y0 = np.clip(cy - side/2, 0, max_h)
    new_x1 = np.clip(cx + side/2, 0, max_w)
    new_y1 = np.clip(cy + side/2, 0, max_h)
    bboxes = np.stack([new_x0, new_y0, new_x1, new_y1], axis=1)
    return bboxes.astype(np.int32)

def vectorized_get_mask_torch(bboxes, hd, wd, device, chunk_size=32):
    N = bboxes.shape[0]
    masks = []
    for i in range(0, N, chunk_size):
        bboxes_chunk = bboxes[i:i+chunk_size]
        n_chunk = bboxes_chunk.shape[0]
        Y = torch.arange(hd, device=device).view(1, hd, 1)
        X = torch.arange(wd, device=device).view(1, 1, wd)
        x0 = bboxes_chunk[:, 0].view(n_chunk, 1, 1)
        y0 = bboxes_chunk[:, 1].view(n_chunk, 1, 1)
        x1 = bboxes_chunk[:, 2].view(n_chunk, 1, 1)
        y1 = bboxes_chunk[:, 3].view(n_chunk, 1, 1)
        mask = ((X >= x0) & (X < x1) & (Y >= y0) & (Y < y1)).to(torch.float32)
        mask = mask.unsqueeze(-1).repeat(1, 1, 1, 3) * 255.0
        masks.append(mask)
    return torch.cat(masks, dim=0)

def vectorized_get_face_contour_torch(face_keypoints, hd, wd, device, chunk_size=32):
    N = face_keypoints.shape[0]
    contours = []
    for i in range(0, N, chunk_size):
        chunk = face_keypoints[i:i+chunk_size]
        n_chunk = chunk.shape[0]
        contour = torch.zeros((n_chunk, hd, wd), device=device)
        x_coords = torch.clamp(chunk[:, :, 0].long(), 0, wd - 1)
        y_coords = torch.clamp(chunk[:, :, 1].long(), 0, hd - 1)
        idx = torch.arange(n_chunk, device=device).unsqueeze(1).expand_as(x_coords)
        contour[idx, y_coords, x_coords] = 1.0
        contour = contour.unsqueeze(-1).repeat(1, 1, 1, 3) * 255.0
        contours.append(contour.cpu())  # move chunk to CPU to free GPU memory
    return torch.cat(contours, dim=0)

def vectorized_get_face_box_torch(face_keypoints, indices, max_w, max_h, scale):
    pts = face_keypoints[:, indices, :2]
    x_min = pts[:, :, 0].min(dim=1)[0]
    y_min = pts[:, :, 1].min(dim=1)[0]
    x_max = pts[:, :, 0].max(dim=1)[0]
    y_max = pts[:, :, 1].max(dim=1)[0]
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    side = torch.max(x_max - x_min, y_max - y_min) * scale
    new_x0 = torch.clamp(cx - side/2, 0, max_w)
    new_y0 = torch.clamp(cy - side/2, 0, max_h)
    new_x1 = torch.clamp(cx + side/2, 0, max_w)
    new_y1 = torch.clamp(cy + side/2, 0, max_h)
    return torch.stack([new_x0, new_y0, new_x1, new_y1], dim=1).long()

           
class DyadicTalking(Dataset):
    def __init__(
        self,
        cfg=None,
        split='train',
    ):
        super().__init__()
        self.cfg = cfg

        vid_meta = []
        for data_meta_path in cfg.data.meta_paths:
            vid_meta.extend(json.load(open(data_meta_path, "r")))
        self.vid_meta = [item for item in vid_meta if item.get("mode") == split]
        self.data_list = self.vid_meta
        self.fps = cfg.model.pose_fps
        self.audio_sr = cfg.model.audio_sr
        self.mean = 0
        self.std = 1
        self.img_size = self.cfg.data.img_size
        self.pixel_transform = transforms.Compose(
            [    
                transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BICUBIC), 
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.pixel_norm = transforms.Compose(
            [   
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
                
            ]
        )
        self.generate_cache = False
        self.face_id = 0
        self.device = None


    @staticmethod
    def normalize(motion, mean, std):
        return (motion - mean) / (std + 1e-7)
    
    @staticmethod
    def inverse_normalize(motion, mean, std):
        # return motion * torch.from_numpy(std).to(motion.device) + torch.from_numpy(mean).to(motion.device)
        return motion * torch.tensor(std).to(motion.device) + torch.tensor(mean).to(motion.device)
    
    def augmentation(self, images, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        
        if isinstance(images, list):
            transformed_images = [transform(Image.fromarray(img))[None] for img in images]
            transformed_images = torch.cat(transformed_images)
            return transformed_images  # (f, c, h, w)
        else:
            return transform(Image.fromarray(images))  # (c, h, w)
    
    def resample_video_to_target_fps(self, video_reader, target_fps, start_idx, end_idx):
        original_fps = video_reader.get_avg_fps()
        target_frame_count = end_idx - start_idx
        # print(target_frame_count, original_fps, target_fps)
        original_frame_count = int(target_frame_count / target_fps * original_fps)
        # print(original_fps, target_frame_count, original_frame_count)
        target_frame_indices = [
            min(int((t / target_fps) * original_fps), original_frame_count - 1)
            for t in range(target_frame_count)
        ]
        target_frame_indices = np.array(target_frame_indices) + start_idx
        video_clip_np = video_reader.get_batch(target_frame_indices).asnumpy()
        return video_clip_np, target_frame_indices
    
    def __len__(self):
        return len(self.data_list)

    def get_hybrid_face_mask_v2_torch(self, face_keypoints_dict, target_frame_indices, face_id, video_clip_np, chunk_size=32):
        device = self.device
        print(f"Using device: {device}")  
        print(f"Available memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB") 
        print(f"Video clip shape: {video_clip_np.shape}")  # (375, 2160, 3840, 3)
        hd, wd, _ = video_clip_np[0].shape

        # Convert face keypoints to tensor and scale to video dimensions
        face_keypoints = torch.from_numpy(face_keypoints_dict[face_id][target_frame_indices].copy())
        face_keypoints[:, :, 0] *= wd
        face_keypoints[:, :, 1] *= hd

        # Define keypoint indices for facial features
        left_eye_indices = torch.tensor([463, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 362], device=device)
        right_eye_indices = torch.tensor([33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7], device=device)
        mouth_indices = torch.tensor([61, 146, 146, 91, 91, 181, 181, 84, 84, 17, 17, 314, 314, 405, 405, 321, 321, 375, 375, 291,
                                    61, 185, 185, 40, 40, 39, 39, 37, 37, 0, 0, 267, 267, 269, 269, 270, 270, 409, 409, 291,
                                    78, 95, 95, 88, 88, 178, 178, 87, 87, 14, 14, 317, 317, 402, 402, 318, 318, 324, 324, 308], device=device)

        # Lists to store processed chunks
        processed_video_list = []
        eye_mouth_mask_list = []

        # Calculate number of chunks
        num_frames = len(target_frame_indices)
        num_chunks = (num_frames + chunk_size - 1) // chunk_size  # Ceiling division

        # Process video in chunks
        for i in range(num_chunks):
            print(f"Processing chunk {i+1}/{num_chunks}")
            start = i * chunk_size
            end = min(start + chunk_size, num_frames)
            chunk_indices = target_frame_indices[start:end]

            # Get face keypoints for the current chunk
            face_keypoints_chunk = face_keypoints[start:end].to(device)

            # Compute bounding boxes for facial features
            l_eye_boxes_chunk = vectorized_get_face_box_torch(face_keypoints_chunk, left_eye_indices, wd, hd, self.cfg.data.eye_bbox_scale)
            r_eye_boxes_chunk = vectorized_get_face_box_torch(face_keypoints_chunk, right_eye_indices, wd, hd, self.cfg.data.eye_bbox_scale)
            mouth_boxes_chunk = vectorized_get_face_box_torch(face_keypoints_chunk, mouth_indices, wd, hd, self.cfg.data.mouth_bbox_scale)

            # Inside your processing loop
            l_eye_masks_chunk = vectorized_get_mask_torch(l_eye_boxes_chunk, hd, wd, device, chunk_size).to(device)
            r_eye_masks_chunk = vectorized_get_mask_torch(r_eye_boxes_chunk, hd, wd, device, chunk_size).to(device)
            mouth_masks_chunk = vectorized_get_mask_torch(mouth_boxes_chunk, hd, wd, device, chunk_size).to(device)

            eye_masks_chunk = torch.where((l_eye_masks_chunk > 0) | (r_eye_masks_chunk > 0), 255, 0).float()
            eye_mouth_mask_chunk = torch.where((mouth_masks_chunk > 0) | (eye_masks_chunk > 0), 255, 0).float()
            eye_mouth_mask_f_chunk = eye_mouth_mask_chunk / 255.0

            face_contour_chunk = vectorized_get_face_contour_torch(face_keypoints_chunk, hd, wd, device, chunk_size).to(device)
            face_contour_f_chunk = face_contour_chunk / 255.0
            contour_without_eyes_chunk = face_contour_f_chunk * (1 - eye_mouth_mask_f_chunk)

            # Load and process the video chunk
            video_chunk_np = video_clip_np[chunk_indices]
            video_chunk = torch.from_numpy(video_chunk_np).to(device).float() / 255.0
            processed_chunk = video_chunk * eye_mouth_mask_f_chunk + contour_without_eyes_chunk
            processed_chunk = torch.clamp(processed_chunk, 0, 1) * 255.0
            print(f"Available memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB") 

            # Move results to CPU
            processed_chunk_cpu = processed_chunk.cpu().byte().numpy()
            eye_mouth_mask_f_cpu = eye_mouth_mask_f_chunk.cpu()

            # Store processed chunks
            processed_video_list.append(processed_chunk_cpu)
            eye_mouth_mask_list.append(eye_mouth_mask_f_cpu)

        # Concatenate all chunks into final outputs
        processed_video = np.concatenate(processed_video_list, axis=0)
        eye_mouth_mask_f = torch.cat(eye_mouth_mask_list, dim=0)

        return eye_mouth_mask_f, processed_video
    
    def get_crop_bbox(self, face_id, bounding_box_dict, video_clip_np, target_frame_indices, video_metadata):
        bounding_box = bounding_box_dict[face_id]
        if isinstance(bounding_box, list):
            bounding_box = np.array(bounding_box)
        check_bbox(bounding_box.reshape(-1, 2, 2))
        hd, wd, _ = video_clip_np[0].shape
    
        ### filtering large translation by union_mask_img
        vid_batch_bbox = bounding_box[target_frame_indices]
        union_bbox_full_video = get_move_area(vid_batch_bbox, wd, hd)
        # union_mask_aspect_ratio = (union_bbox_full_video[2]-union_bbox_full_video[0])/(union_bbox_full_video[3]-union_bbox_full_video[1])
        # if union_mask_aspect_ratio > 1.2: 
        #     raise Exception("Large translation")

        # # get frame-wise face bounding box 
        # batch_bbox = bounding_box[target_frame_indices]
        # frame_union_mask_area = (batch_bbox[:, 2]-batch_bbox[:, 0])*(batch_bbox[:, 3]-batch_bbox[:, 1])
        # frame_union_mask_area_ratio= frame_union_mask_area / frame_union_mask_area[0]
        # if (frame_union_mask_area_ratio < 0.82).any():
        #     raise Exception('dynamic camera')

        # generate human-centered bounding box 
        union_bbox_cond = union_bbox_full_video
        width = union_bbox_cond[2] - union_bbox_cond[0]
        height = union_bbox_cond[3] - union_bbox_cond[1]
        if isinstance(self.cfg.data.union_bbox_scale, float):
            union_bbox_scale = self.cfg.data.union_bbox_scale
        else:
            if self.cfg.data.union_bbox_scale[0] == self.cfg.data.union_bbox_scale[1]:
                union_bbox_scale = self.cfg.data.union_bbox_scale[0]
            else:
                union_bbox_scale = np.random.uniform(*self.cfg.data.union_bbox_scale)
    
        if max(width, height) <= self.cfg.data.img_size[0]:
            max_size=self.cfg.data.img_size[0]
        else:
            max_size = int(max(width, height) * union_bbox_scale)
        
        # max_size = int(max(width, height) * union_bbox_scale)
        center_x = (union_bbox_cond[0] + union_bbox_cond[2]) / 2
        center_y = (union_bbox_cond[1] + union_bbox_cond[3]) / 2
        center = [int(center_y), int(center_x)]
        crop_bbox = generate_crop_bounding_box(hd, wd, center, max_size)
        
        if self.cfg.data.filter_hand_videos and video_metadata["frame_data"].get("bounding_box_hand", None) is not None:
            hands_boxes_org = video_metadata["frame_data"]["bounding_box_hand"][target_frame_indices]
            ## check whether the hands are outside the face bounding box
            if len(hands_boxes_org.shape) == 1 and (hands_boxes_org==-1).all():
                pass
            else:
                hands_boxes = hands_boxes_org.reshape(-1, 4)
                bbox1_x_min = hands_boxes[:, 0]
                bbox1_y_min = hands_boxes[:, 1]
                bbox1_x_max = hands_boxes[:, 2]
                bbox1_y_max = hands_boxes[:, 3]
                bbox2_x_min, bbox2_y_min, bbox2_x_max, bbox2_y_max = crop_bbox
                condition = (
                    (bbox1_x_max < bbox2_x_min) |  # bbox1 is on the left side of bbox2
                    (bbox1_x_min > bbox2_x_max) |  # bbox1 is on the right side of bbox2
                    (bbox1_y_max < bbox2_y_min) |  # bbox1 is on the top side of bbox2
                    (bbox1_y_min > bbox2_y_max)    # bbox1 is on the bottom side of bbox2
                )
                if not condition.all():
                    raise Exception("0, Hands in video!")
                
        return crop_bbox, center, max_size, hd, wd
    
    def get_item(self, item):
        print("get item, ", item)
        face_id = self.face_id
        data_item = self.data_list[item]
        print(f"checking /mnt/weka/lhy_workspace/training_data_hy/{args.dataset_name}/cache_latent_v1/{data_item['video_id']}_{face_id}.npz ...")
        if os.path.exists(f"/mnt/weka/lhy_workspace/training_data_hy/{args.dataset_name}/cache_latent_v1/{data_item['video_id']}_{face_id}.npz"):
            print("cache exists")
            return {"video_id": [-1]}
        # meta information
        video_metadata = np.load(data_item["metadata_path"], allow_pickle=True)["arr_0"].tolist()
        if self.generate_cache:
            sdx, edx = 0, video_metadata["frame_count"]
            face_id = self.face_id # random.choice([0, 1])
        else:
            sdx, edx = data_item["start_idx"], data_item["end_idx"]
            face_id = random.choice([0, 1]) 
        print("face_id: ", face_id)
        
        if self.cfg.data.get_video:
            import time
            start_time = time.time()
            video_reader = VideoReader(data_item["resampled_video_path"]) # VideoReader(data_item["original_video_path"])
            # step 1: adjust fps
            video_clip_np, target_frame_indices = self.resample_video_to_target_fps(video_reader, self.fps, sdx, edx)
            print("read video time: ", time.time()-start_time)
            start_time = time.time()
            video_clip_np4mask = video_clip_np.copy()   
            eye_mouth_mask, masked_video_clip_np = self.get_hybrid_face_mask_v2_torch(video_metadata["frame_data"]["keypoints"], target_frame_indices, face_id, video_clip_np4mask)
            print("get mask time: ", time.time()-start_time)
            start_time = time.time()
            bounding_box_dict = video_metadata["frame_data"]["bounding_box"]
            crop_bbox, center, max_size, hd, wd = self.get_crop_bbox(face_id, bounding_box_dict, video_clip_np, target_frame_indices, video_metadata)
            print("get crop bbox time: ", time.time()-start_time)
            start_time = time.time()
            vid_pil_image_list = [crop_from_bbox(img, center, crop_bbox, size=max_size) for img in video_clip_np]
            masked_pil_img_list = [crop_from_bbox(img, center, crop_bbox, size=max_size) for img in masked_video_clip_np]
            print("crop image time: ", time.time()-start_time)
            start_time = time.time()
            vid_tensor = torch.from_numpy(np.stack(vid_pil_image_list, axis=0)).permute(0,3,1,2).float() / 255.0 * 2 - 1
            masked_tensor = torch.from_numpy(np.stack(masked_pil_img_list, axis=0)).permute(0,3,1,2).float() / 255.0 * 2 - 1
            pixel_values_vid = F.interpolate(vid_tensor, size=tuple(self.img_size), mode='bilinear', align_corners=False)
            masked_pixel_values_vid = F.interpolate(masked_tensor, size=tuple(self.img_size), mode='bilinear', align_corners=False)
            print("Time: ", time.time()-start_time)
            ref_img = pixel_values_vid[0].clone()
            # todo step 4: flip augmentation
            cropped_face_tensor = pixel_values_vid 
            cropped_hybrid_face_tensor = masked_pixel_values_vid
            ref_face_tensor = ref_img # without mask
        else: 
            cropped_face_tensor, cropped_hybrid_face_tensor, ref_face_tensor = -1, -1, -1

        # audio 
        if self.cfg.data.get_audio:
            audio_self_path, audio_other_path = data_item["audio_self_path"], data_item["audio_other_path"]
            if face_id == 1:
                audio_self_path, audio_other_path = audio_other_path, audio_self_path

            audio, _ = librosa.load(audio_self_path, sr=self.audio_sr)
            sdx_audio = sdx * int((1 / self.cfg.model.pose_fps) * self.audio_sr)
            edx_audio = edx * int((1 / self.cfg.model.pose_fps) * self.audio_sr)
            audio = audio[sdx_audio:edx_audio]
            audio_tensor = torch.from_numpy(audio).float()
            
            audio_other, _ = librosa.load(audio_other_path, sr=self.audio_sr)
            audio_other = audio_other[sdx_audio:edx_audio]
            audio_other_tensor = torch.from_numpy(audio_other).float()
            diff_audio = edx_audio - sdx_audio - audio_tensor.shape[0]
            if diff_audio != 0:
                print("padding audio", diff_audio)
                if diff_audio > 0:
                    audio_tensor = torch.cat([audio_tensor, audio_tensor[-diff_audio:]], dim=0)
                else:
                    audio_tensor = audio_tensor[:self.cfg.model.pose_length]
            diff_audio_other = edx_audio - sdx_audio - audio_other_tensor.shape[0]
            if diff_audio_other != 0:
                print("padding audio_other", diff_audio_other)
                if diff_audio_other > 0:
                    audio_other_tensor = torch.cat([audio_other_tensor, audio_other_tensor[-diff_audio_other:]], dim=0)
                else:
                    audio_other_tensor = audio_other_tensor[:self.cfg.model.pose_length]
        else:
            audio_tensor, audio_other_tensor = -1, -1

        # motion latent 
        if self.cfg.data.get_motion:
            motion_self_path, motion_other_path = data_item["motion_self_path"], data_item["motion_other_path"]
            if face_id == 1:
                motion_self_path, motion_other_path = motion_other_path, motion_self_path

            motion_dict = np.load(motion_self_path, allow_pickle=True)
            motion = motion_dict["random_data"][sdx:edx]
            # print(motion_dict["random_data"].shape, sdx, edx)
            # motion = self.normalize(motion, self.mean, self.std)
            motion_dict_other = np.load(motion_other_path, allow_pickle=True)
            motion_other = motion_dict_other["random_data"][sdx:edx]
            
            if np.random.rand() > self.cfg.data.random_mix:
                length = data_item["frames"] - (edx-sdx) - 1
                ref_sdx = np.random.randint(0, length)
                ref_edx = ref_sdx + (edx-sdx)
                ref_motion = motion_dict["random_data"][ref_sdx:ref_edx]
                ref_motion_other = motion_dict_other["random_data"][ref_sdx:ref_edx]
            else:
                ref_motion = motion
                ref_motion_other = motion_other
        
            motion_tensor = torch.from_numpy(motion).float()
            ref_motion_tensor = torch.from_numpy(ref_motion).float()
            motion_other_tensor = torch.from_numpy(motion_other).float()
            ref_motion_other_tensor = torch.from_numpy(ref_motion_other).float()
            # padding
            diff_motion = self.cfg.model.pose_length - motion_tensor.shape[0]
            if diff_motion != 0:
                print("padding motion", diff_motion)
                if diff_motion > 0:
                    motion_tensor = torch.cat([motion_tensor, motion_tensor[-diff_motion:]], dim=0)
                    ref_motion_tensor = torch.cat([ref_motion_tensor, ref_motion_tensor[-diff_motion:]], dim=0)
                else:
                    motion_tensor = motion_tensor[:self.cfg.model.pose_length]
                    ref_motion_tensor = ref_motion_tensor[:self.cfg.model.pose_length]
            diff_motion_other = self.cfg.model.pose_length - motion_other_tensor.shape[0]
            if diff_motion_other != 0:
                print("padding motion_other", diff_motion_other)
                if diff_motion_other > 0:
                    motion_other_tensor = torch.cat([motion_other_tensor, motion_other_tensor[-diff_motion_other:]], dim=0)
                    ref_motion_other_tensor = torch.cat([ref_motion_other_tensor, ref_motion_other_tensor[-diff_motion_other:]], dim=0)
                else:
                    motion_other_tensor = motion_other_tensor[:self.cfg.model.pose_length]
                    ref_motion_other_tensor = ref_motion_other_tensor[:self.cfg.model.pose_length]
            # print(motion_tensor.shape[0]/30, audio_tensor.shape[0]/16000)
        else:
            motion_tensor, ref_motion_tensor, motion_other_tensor, ref_motion_other_tensor = -1, -1, -1, -1
        print(item, data_item["video_id"])
        return dict(
            motion_latent=motion_tensor,
            audio=audio_tensor, 
            style_latent=ref_motion_tensor,
            motion_latent_other=motion_other_tensor,
            audio_other=audio_other_tensor,
            style_latent_other=ref_motion_other_tensor,
            face_video=cropped_face_tensor,
            hybrid_face_video=cropped_hybrid_face_tensor,
            ref_face_img=ref_face_tensor,
            video_id=data_item["video_id"],
        )

    def __getitem__(self, item):
        # return self.get_item(item)
        try:
            return self.get_item(item)
        except:
            return {"video_id": [-1]}
        
def visualize_video(data, output_filename):
    ref_img = data['ref_face_img'][0]          # [3, 512, 512]
    face_video = data['face_video'][0]           # [64, 3, 512, 512]
    hybrid_video = data['hybrid_face_video'][0]  # [64, 3, 512, 512]

    num_frames = face_video.shape[0]
    ref_video = ref_img.unsqueeze(0).repeat(num_frames, 1, 1, 1)
    ref_video_np = ref_video.permute(0, 2, 3, 1).cpu().numpy() * 0.5 + 0.5
    face_video_np = face_video.permute(0, 2, 3, 1).cpu().numpy() * 0.5 + 0.5
    hybrid_video_np = hybrid_video.permute(0, 2, 3, 1).cpu().numpy() * 0.5 + 0.5
    
    combined_video = np.concatenate([ref_video_np, face_video_np, hybrid_video_np], axis=2)
    combined_video = combined_video * 255
    combined_video = np.clip(combined_video, 0, 255).astype(np.uint8)
    imageio.mimwrite(output_filename, combined_video, fps=25)
    
def visualize_single(video_data, output_filename):
    video_data = video_data.permute(0, 2, 3, 1).cpu().numpy() * 0.5 + 0.5
    video_data = video_data * 255
    video_data = np.clip(video_data, 0, 255).astype(np.uint8)
    imageio.mimwrite(output_filename, video_data, fps=25)
    
def get_motion_latent(video_path, motion_encoder, device):
    frames_video = VideoReader(video_path)
    frames_np = frames_video.get_batch(range(len(frames_video))).asnumpy()
    frames_np = frames_np.astype(np.float32) / 255.
    frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).to(device)
    frames_tensor = (frames_tensor - 0.5) / 0.5
    # print(f"Vectorized transform took {time.time() - start_time:.2f} seconds")

    chunk_size = 250
    all_latents = []
    with torch.no_grad():
        for i in range(0, frames_tensor.shape[0], chunk_size):
            batch = frames_tensor[i:i+chunk_size]
            latents_chunk = motion_encoder(batch)
            all_latents.append(latents_chunk[0].detach().cpu().numpy())
    # print(f"Motion encoding took {time.time() - start_time:.2f} seconds")
    return np.concatenate(all_latents, axis=0)
    
    
if __name__ == "__main__":
    import argparse
    from tqdm import tqdm
    import os
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default="xxx.json")
    parser.add_argument("--dataset_name", type=str, default="xxx")
    parser.add_argument("--split_idx", type=int, default=0)
    parser.add_argument("--cfg", type=str, default="/home/weili/haiyang/DyanicTalking/configs/dyadic_data_process.yaml")
    parser.add_argument("--face_id", type=int, default=0)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()
    
    device = torch.device(f"cuda:{args.gpu_id}")
    # train_test_spilt
    generate_cache = True
    face_id = args.face_id
    cfg = OmegaConf.load(args.cfg)
    cfg.data.meta_paths = [args.json_path]
    dataset = DyadicTalking(cfg, split=f"train_{args.split_idx}")
    dataset.generate_cache = generate_cache
    dataset.face_id = face_id
    dataset.device = device
    dataloader= torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True) #  
    print("logging step 1")
    
    from utils import instantiate
    config = OmegaConf.load("/home/weili/haiyang/PantoMatrix/datasets/motion_gen_train.yaml")
    motion_encoder = instantiate(config.model.motion_encoder)
    # params = torch.load(config.model.motion_encoder_path)["state_dict"]
    params = torch.load(config.model.motion_encoder_path, map_location=torch.device('cpu'))["state_dict"]
    adjusted_dict = {k.replace("motion_encoder.", ""): v for k, v in params.items() if k.startswith("motion_encoder.")}
    motion_encoder.load_state_dict(adjusted_dict)
    motion_encoder = motion_encoder.to(device)
    print("logging step 2")
    print(len(dataloader))
    
    fail_count = 0
    for idx, data in tqdm(enumerate(dataloader)):
        if data["video_id"][0] == -1:
            fail_count += 1
            print(f"Failed {fail_count} times")
            continue
        print(f"Processing {idx}th video: {data['video_id'][0]}")
        # if os.path.exists(f"/mnt/weka/lhy_workspace/training_data_hy/{args.dataset_name}/cache_latent_v1/{data['video_id'][0]}_{face_id}.npz"):
        #     continue
        face_video = data['face_video'][0]           # [64, 3, 512, 512]
        hybrid_video = data['hybrid_face_video'][0]  # [64, 3, 512, 512]
        
        os.makedirs(f"/mnt/weka/lhy_workspace/training_data_hy/{args.dataset_name}/cache_ori_v1", exist_ok=True)
        os.makedirs(f"/mnt/weka/lhy_workspace/training_data_hy/{args.dataset_name}/cache_facedet_v1", exist_ok=True)
        os.makedirs(f"/mnt/weka/lhy_workspace/training_data_hy/{args.dataset_name}/cache_latent_v1", exist_ok=True)
        visualize_single(face_video, f"/mnt/weka/lhy_workspace/training_data_hy/{args.dataset_name}/cache_ori_v1/{data['video_id'][0]}_{face_id}.mp4")
        visualize_single(hybrid_video, f"/mnt/weka/lhy_workspace/training_data_hy/{args.dataset_name}/cache_facedet_v1/{data['video_id'][0]}_{face_id}.mp4")
        motion_latent = get_motion_latent(f"/mnt/weka/lhy_workspace/training_data_hy/{args.dataset_name}/cache_facedet_v1/{data['video_id'][0]}_{face_id}.mp4", motion_encoder, device)
        np.savez(f"/mnt/weka/lhy_workspace/training_data_hy/{args.dataset_name}/cache_latent_v1/{data['video_id'][0]}_{face_id}.npz", random_data=motion_latent)
       