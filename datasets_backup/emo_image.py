import os
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from decord import VideoReader
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.v2.functional import pad
from transformers import CLIPImageProcessor


def crop(img_array, bbox):
    bbox_x0, bbox_y0, bbox_x1, bbox_y1 = [int(ii) for ii in bbox]
    img_crop = img_array[bbox_y0:bbox_y1, bbox_x0:bbox_x1, :]
    return img_crop


def tonp(x):
    return x.detach().cpu().numpy()


def crop_and_resize(images, new_size, original_size, start):
    cropped_image = transforms.functional.crop(images, start[1], start[0], new_size, new_size)
    resized_image = transforms.Resize((original_size, original_size))(cropped_image)
    return resized_image


def resize_and_pad(images, new_size, padding, pixel_trans=1):
    resized_image = transforms.Resize((new_size, new_size))(images)
    if pixel_trans:
        padded_image = pad(resized_image, padding, fill=0.0, padding_mode="edge")
    else:
        padded_image = pad(resized_image, padding, fill=0.0, padding_mode="constant")
    return padded_image


def scale_bbox(bbox, h, w, scale=1.8):
    sw = (bbox[2] - bbox[0]) / 2
    sh = (bbox[3] - bbox[1]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    cx = (bbox[0] + bbox[2]) / 2
    sw *= scale
    sh *= scale
    scale_bbox = [cx - sw, cy - sh, cx + sw, cy + sh]
    scale_bbox[0] = np.clip(scale_bbox[0], 0, w)
    scale_bbox[2] = np.clip(scale_bbox[2], 0, w)
    scale_bbox[1] = np.clip(scale_bbox[1], 0, h)
    scale_bbox[3] = np.clip(scale_bbox[3], 0, h)
    return scale_bbox


def mediapipe2s3fd(bbox, extension_factor=0.1):
    bbox_height = bbox[3] - bbox[1]
    bbox_scale_up_value = extension_factor * bbox_height
    bbox[1] = int(max(0, bbox[1] - bbox_scale_up_value))
    return bbox


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


def get_move_area(bbox, fw, fh):
    move_area_bbox = [
        bbox[:, 0].min(),
        bbox[:, 1].min(),
        bbox[:, 2].max(),
        bbox[:, 3].max(),
    ]

    if move_area_bbox[0] < 0:
        move_area_bbox[0] = 0
    if move_area_bbox[1] < 0:
        move_area_bbox[1] = 0
    if move_area_bbox[2] > fw:
        move_area_bbox[2] = fw
    if move_area_bbox[3] > fh:
        move_area_bbox[3] = fh
    return move_area_bbox


def crop_np(img_array, bbox):
    bbox_x0, bbox_y0, bbox_x1, bbox_y1 = [int(ii) for ii in bbox]
    img_crop = img_array[bbox_y0:bbox_y1, bbox_x0:bbox_x1, :]
    return img_crop


def xyxy_to_xys(det):
    s = max((det[3] - det[1]), (det[2] - det[0])) / 2
    y = (det[1] + det[3]) / 2  # crop center x
    x = (det[0] + det[2]) / 2  # crop center y
    return [x, y, s]


def xys_to_xyxy(x, y, s, scale=3.0):
    s *= scale
    return np.stack([x - s, y - s, x + s, y + s], axis=-1)


def get_rand_s(width, height, bbox):
    bbox_s = bbox[2]
    max_s = min(min(bbox[0], width - bbox[0]) / (bbox_s), min(bbox[1], height - bbox[1]) / (bbox_s))
    if max_s < 1:
        max_s = 1
    if max_s > 2.5:
        max_s = 2.5

    s = max_s
    return s


def get_scale_bbox(wd, hd, union_bbox):
    union_bbox_xys = xyxy_to_xys(union_bbox)
    scale = get_rand_s(wd, hd, union_bbox_xys)
    scale_bbox = xys_to_xyxy(*union_bbox_xys, scale=scale)
    return scale_bbox


class random_crop_with_bbox:
    def __init__(self, scale_bbox, transform) -> None:
        self.scale_bbox = scale_bbox
        self.transform = transform

    def __call__(self, img):
        cropped_img = img.crop(self.scale_bbox)
        return self.transform(cropped_img)


class ImageDataset(Dataset):
    def __init__(
        self,
        img_size,
        img_scale,
        img_ratio,
        zoom_out_ratio=None,
        drop_ratio=0.1,
        sample_margin=30,
        cfg=None,
        save_gt=False,
    ):
        super().__init__()
        self.cfg = cfg
        self.save_gt = save_gt

        self.img_size = img_size
        self.img_scale = img_scale
        self.img_ratio = img_ratio
        self.zoom_out_ratio = zoom_out_ratio
        self.sample_margin = sample_margin

        if hasattr(self.cfg.data, "meta_paths") and hasattr(self.cfg.data, "root_video_paths"):
            # Alex dataloader
            self.data_root_paths = self.cfg.data.root_video_paths
            self.data_meta_paths = self.cfg.data.meta_paths
            assert len(self.data_root_paths) == len(self.data_meta_paths)

            # Paths to all of the metadata
            self.idx_to_range = {}
            prev_idx = 0
            vid_meta = []
            dset_sample_rate = []
            for idx, meta_path in enumerate(self.data_meta_paths):
                if os.path.exists(Path(meta_path).parent / "index.npz"):
                    # [metadata_name, loading_failure, min_offset, max_conf, audio_exists, audio_feature_exists]
                    index = np.load(str(Path(meta_path).parent / "index.npz"))["arr_0"]
                    valid = 1 - index[:, 1].astype(int)

                    stem_dir = list(index[valid.astype(np.bool_), 0])
                else:
                    stem_dir = os.listdir(meta_path)

                print(f"{meta_path}: {len(stem_dir)} samples. Is idx {idx}")
                dset_sample_rate.append(len(stem_dir))
                self.idx_to_range[idx] = (prev_idx, prev_idx + len(stem_dir))
                prev_idx = prev_idx + len(stem_dir)
                vid_meta += [(idx, os.path.join(meta_path, video_stem)) for video_stem in stem_dir]

            dset_sample_rate = np.array(dset_sample_rate) / len(vid_meta)
            self.dset_sample_rate = dset_sample_rate

            self.meta_dirs = vid_meta
            print(f"{len(self.meta_dirs)=}")

            used_mediapipe_masks = self.cfg.data.get("mediapipe_mask", None)
            if used_mediapipe_masks is None:
                used_mediapipe_masks = [False for _ in range(len(self.data_meta_paths))]
            self.used_mediapipe_masks = used_mediapipe_masks
        elif hasattr(self.cfg.data, "dataset_file_path"):
            # pq dataset DATALOADER
            self.dataset_file_path = self.cfg.data["dataset_file_path"]
            assert os.path.exists(Path(self.dataset_file_path)), "Dataset file not found!"

            index = np.load(self.dataset_file_path, allow_pickle=True)["arr_0"]

            self.data_root_paths = sorted(list(set([str(Path(entry[0]) / "videos_resampled") for entry in index])))
            vid_meta = [
                (
                    self.data_root_paths.index(str(Path(entry[0]) / "videos_resampled")),
                    str(Path(entry[0]) / "metadata" / entry[1]),
                )
                for entry in index
            ]
            self.meta_dirs = vid_meta

            # dset_sample_rate is not defined, so no weighted sampling on the dataset. This should be done
            # in construction of the dataset
            used_mediapipe_masks = self.cfg.data.get("mediapipe_mask", True)
            used_mediapipe_masks = [used_mediapipe_masks for _ in range(len(self.data_root_paths))]
            self.used_mediapipe_masks = used_mediapipe_masks
        else:
            # OLD dataloader
            vid_meta = []
            self.data_meta_paths = self.cfg.data.meta_paths
            for data_meta_path in self.data_meta_paths:
                vid_meta += torch.load(data_meta_path)
            self.vid_meta = vid_meta
            print(f"{len(self.vid_meta)=}")
        self.meta_dirs = self.meta_dirs * int(1000000 // len(self.meta_dirs))
        self.clip_image_processor = CLIPImageProcessor()

        min_side_len = min(self.img_size)
        zoom_out_transforms = []
        zoom_out_cond_transforms = []
        if self.zoom_out_ratio is not None:
            # Works for square or widescreen, since the min_size and max_size corresponds to
            # the smaller dimension of the image. Then, the random crop pads ON ALL SIDES the difference
            # between the smaller resized image and the size it has to be, and then randomly crops this
            # This leads to less zoomed out results from observing but overall look pretty similar
            zoom_out_transforms = [
                transforms.RandomResize(
                    min_size=int(self.zoom_out_ratio[0] * min_side_len),
                    max_size=int(self.zoom_out_ratio[1] * min_side_len),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                transforms.RandomCrop(
                    self.img_size,
                    pad_if_needed=True,
                    padding_mode="edge",
                ),
            ]
            zoom_out_cond_transforms = [
                transforms.RandomResize(
                    min_size=int(self.zoom_out_ratio[0] * min_side_len),
                    max_size=int(self.zoom_out_ratio[1] * min_side_len),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                transforms.RandomCrop(
                    self.img_size,
                    pad_if_needed=True,
                    fill=0,
                    padding_mode="constant",
                ),
            ]

        aspect_ratio = self.img_size[1] / self.img_size[0]
        self.transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomChoice(
                    [
                        # Either get a square crop and apply zoom-out effect
                        transforms.Compose(
                            [
                                transforms.RandomResizedCrop(
                                    (min_side_len, min_side_len),
                                    scale=self.img_scale,
                                    ratio=self.img_ratio,
                                    interpolation=transforms.InterpolationMode.BILINEAR,
                                    antialias=True,
                                ),
                                *zoom_out_transforms,
                            ]
                        ),
                        # Or get a crop with the final resolution (mainly zooms into the face)
                        transforms.RandomResizedCrop(
                            self.img_size,
                            scale=self.img_scale,
                            ratio=[ratio * aspect_ratio for ratio in self.img_ratio],
                            interpolation=transforms.InterpolationMode.BILINEAR,
                            antialias=True,
                        ),
                    ],
                    p=[0.5, 0.5],
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.cond_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomChoice(
                    [
                        # Either get a square crop and apply zoom-out effect
                        transforms.Compose(
                            [
                                transforms.RandomResizedCrop(
                                    (min_side_len, min_side_len),
                                    scale=self.img_scale,
                                    ratio=self.img_ratio,
                                    interpolation=transforms.InterpolationMode.BILINEAR,
                                    antialias=True,
                                ),
                                *zoom_out_cond_transforms,
                            ]
                        ),
                        # Or get a crop with the final resolution (mainly zooms into the face)
                        transforms.RandomResizedCrop(
                            self.img_size,
                            scale=self.img_scale,
                            ratio=[ratio * aspect_ratio for ratio in self.img_ratio],
                            interpolation=transforms.InterpolationMode.BILINEAR,
                            antialias=True,
                        ),
                    ],
                    p=[0.5, 0.5],
                ),
                transforms.ToTensor(),
            ]
        )

        self.drop_ratio = drop_ratio

    def augmentation(self, image, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        return transform(image)

    def __getitem__(self, index):
        return self.get_wo_error(index)

    def get_wo_error(self, index):
        try:
            return self.get_item(index)
        except Exception as e:
            import traceback

            traceback.print_exc()
            return self.get_wo_error((index + 1) % self.__len__())

    def __len__(self):
        return len(self.vid_meta)


class EmoDataset(ImageDataset):
    def get_batch_indices(self, center_idx, batch_size, vid_length):
        """
        Get a range of indices centered around center_idx with length batch_size,
        padding evenly on both sides or unevenly if necessary.
        """
        half_batch = batch_size // 2
        start_idx = max(0, center_idx - half_batch)
        end_idx = min(vid_length, center_idx + half_batch + (batch_size % 2))

        if end_idx - start_idx < batch_size:
            # If we're near the edges, adjust the range
            if start_idx == 0:
                end_idx = min(vid_length, batch_size)
            elif end_idx == vid_length:
                start_idx = max(0, vid_length - batch_size)

        return list(range(start_idx, end_idx))

    def get_item(self, index):
        # Get video path and metadata

        video_dset_idx, video_meta_dir = self.meta_dirs[index]
        video_metadata = dict(np.load(meta_path := ((Path(video_meta_dir) / "metadata.npz")), allow_pickle=True))[
            "arr_0"
        ].item()
        video_path = str(Path(self.data_root_paths[video_dset_idx]) / video_metadata["path_to_video"])

        # Get union bbox of full video, normally used to crop the video. If converted data, this should be none
        # since the video is already cropped
        union_bbox = video_metadata.get("bounding_box_union", "pre_processed")
        vid_length = video_metadata.get("frame_count", None)

        # Read video, vid_length is None if this is a converted data
        video_reader = VideoReader(video_path)
        if vid_length is not None:
            assert len(video_reader) == vid_length, "Something went wrong in processing"
        else:
            # Converted dataset
            vid_length = len(video_reader)

        self.union_bbox_as_stage2 = self.cfg.data.get("union_bbox_as_stage2", False)
        self.n_sample_frames = self.cfg.data.get("n_sample_frames", 28)

        if self.union_bbox_as_stage2:
            target_fps = self.cfg.data.train_fps
            original_fps = video_reader.get_avg_fps()

        start_margin = self.cfg.data.get("start_margin", 0)
        margin = min(self.sample_margin, vid_length)

        # Get bounding boxes: select face + sample id
        bounding_box_dict = video_metadata["frame_data"]["bounding_box"]

        def valid_bbox(bbox):
            valid_bbox = True
            try:
                if np.array(bbox).min() < 0:
                    valid_bbox = False
            except:
                valid_bbox = False
            return valid_bbox

        face_id = -1
        valid_face_id = {}
        for potential_face_id in bounding_box_dict.keys():
            bboxes = bounding_box_dict[potential_face_id]
            bbox_valid = [idx for idx, bbox in enumerate(bboxes) if valid_bbox(bbox)]

            if len(bbox_valid) > margin:  # Heuristic, need margin length of masks to see
                valid_face_id[potential_face_id] = bbox_valid

        if len(valid_face_id.keys()) == 0:
            raise Exception("Invalid video sampled")
        face_id = random.choice(list(valid_face_id.keys()))
        valid_bbox_idxs = valid_face_id[face_id]

        # Sample two frames and their individual bbox, try to sample frames not too close
        # to each other

        valid_ref_idxs = [idx for idx in valid_bbox_idxs if idx > start_margin]
        self.hard_trunk = self.cfg.data.get("hard_trunk", None)

        if self.hard_trunk is not None:
            valid_ref_idxs = [idx for idx in valid_ref_idxs if idx % self.hard_trunk == 0]

        ref_img_idx = random.choice(valid_ref_idxs)

        valid_tgt_idxs = [idx for idx in valid_bbox_idxs if np.abs(ref_img_idx - idx) > margin]
        if len(valid_tgt_idxs) == 0:
            tgt_img_idx = random.choice(valid_bbox_idxs)
        else:
            tgt_img_idx = random.choice(valid_tgt_idxs)

        bounding_box = bounding_box_dict[face_id]
        if isinstance(bounding_box, list):
            bounding_box = np.array(bounding_box)

        # Get images
        ref_img = video_reader[ref_img_idx].asnumpy()
        tgt_img = video_reader[tgt_img_idx].asnumpy()

        # Create small mask conditioning based on the per-frame bbox
        hd, wd, _ = tgt_img.shape
        if hasattr(self.cfg.data.train_mask_scale, "__iter__"):
            train_mask_scale = np.random.uniform(
                low=self.cfg.data.train_mask_scale[0], high=self.cfg.data.train_mask_scale[1]
            )
        else:
            train_mask_scale = self.cfg.data.train_mask_scale

        if self.union_bbox_as_stage2:
            batch_tgt_frames = int(self.n_sample_frames / target_fps * original_fps)
            batch_tgt_idx = self.get_batch_indices(tgt_img_idx, batch_tgt_frames, vid_length)
        else:
            batch_tgt_idx = [tgt_img_idx]

        # CONVERT MEDIAPIPE 2 S3FD IF WAS PROCESSED WITH MEDIAPIPE
        hd, wd, _ = video_reader[0].shape
        batch_bbox = bounding_box[batch_tgt_idx]
        bbox_tgt = get_move_area(batch_bbox, wd, hd)

        if self.cfg.data.get("convert_mediapipe_to_s3fd", False) and self.used_mediapipe_masks[video_dset_idx]:
            bbox_tgt = mediapipe2s3fd(bbox_tgt)

        tgt_pose = get_mask(bbox_tgt, hd, wd, scale=train_mask_scale)
        tgt_pose = np.array(tgt_pose)

        # Crop and resize based on the union bbox, if it is present
        if union_bbox is None or union_bbox == "pre_processed":
            union_bbox_scaled = "pre_processed"
        else:
            union_bbox_scaled = scale_bbox(
                np.array(union_bbox[face_id]).reshape(-1).tolist(), hd, wd, scale=self.cfg.data.union_bbox_scale
            )

            ref_img = crop(ref_img, union_bbox_scaled)
            tgt_img = crop(tgt_img, union_bbox_scaled)
            tgt_pose = crop(tgt_pose, union_bbox_scaled)

        # Convert to PIL
        ref_img_pil = Image.fromarray(ref_img)
        tgt_img_pil = Image.fromarray(tgt_img)
        tgt_pose_pil = Image.fromarray(tgt_pose.astype(np.uint8))

        state = torch.get_rng_state()
        tgt_img = self.augmentation(tgt_img_pil, self.transform, state)
        tgt_pose_img = self.augmentation(tgt_pose_pil, self.cond_transform, state)
        ref_img_vae = self.augmentation(ref_img_pil, self.transform, state)

        tgt_bbox = torch.nn.functional.interpolate(tgt_pose_img[None], scale_factor=1 / 8)[0]
        sample = dict(
            video_dir=video_path,
            img=tgt_img,
            tgt_pose=tgt_pose_img,
            tgt_bbox=tgt_bbox,
            ref_img=ref_img_vae,
        )

        if self.cfg.get("use_insightface_emb", False):
            face_emb = self.face_id.get_from_bbox(
                bboxes=bounding_box[ref_img_idx][None, ...], frame_bgr=np.array(ref_img_pil)[..., ::-1]
            )  # (512,)
            sample["face_emb"] = torch.from_numpy(face_emb)

        if self.save_gt:
            save_dir = "./debug_outputs/test_alex_dataloader_stage1"
            os.makedirs(save_dir, exist_ok=True)
            save_path = f"{save_dir}/idx{index}.png"
            import cv2
            import imageio

            ref_np = ((ref_img_vae.permute(1, 2, 0).numpy() + 1.0) * 127.5).astype(np.uint8)
            pose_np = (tgt_pose_img.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            tgt_np = ((tgt_img.permute(1, 2, 0).numpy() + 1.0) * 127.5).astype(np.uint8)
            left_up_x, left_up_y, x2, y2 = list(map(int, bounding_box[tgt_img_idx]))
            # Blend ref_np with tgt_np
            ref_blend = cv2.addWeighted(ref_np, 0.5, tgt_np, 0.5, 0)

            # Blend pose_np with tgt_np
            pose_blend = cv2.addWeighted(pose_np, 0.5, tgt_np, 0.5, 0)

            # Concatenate all 5 images
            concat_im = np.concatenate([ref_np, ref_blend, tgt_np, pose_blend, pose_np], axis=1)
            imageio.imwrite(save_path, concat_im)

        return sample

    def get_wo_error(self, index):
        try:
            if hasattr(self, "dset_sample_rate"):
                sampled_index = int(np.random.choice(a=len(self.dset_sample_rate), size=1, p=self.dset_sample_rate))
                idx_range = self.idx_to_range[sampled_index]
                index = random.randint(idx_range[0], idx_range[1] - 1)
            return self.get_item(index)
        except Exception as e:
            _, video_meta_dir = self.meta_dirs[index]
            return self.get_wo_error((index + 1) % self.__len__())

    def __len__(self):
        return len(self.meta_dirs)


def test_alex_dataloader():
    """
    RUN_FUNC=test_alex_dataloader python src/dataset/emo_image.py
    """
    cfg = OmegaConf.load("configs/train/emo_stage1_lossamp.yaml")
    train_dataset = EmoDataset(
        img_size=(cfg.data.train_height, cfg.data.train_width),
        img_scale=(0.9, 1.0),
        img_ratio=cfg.data.aspect_ratio_range,
        zoom_out_ratio=cfg.data.get("zoom_out_ratio", None),
        sample_margin=cfg.data.sample_margin,
        cfg=cfg,
        save_gt=True,
    )
    num_workers = 0
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.data.train_bs, shuffle=True, num_workers=num_workers
    )
    train_iterator = iter(train_dataloader)
    for i in range(100):
        print(f"{i=}")
        batch = next(train_iterator)


if __name__ == "__main__":
    RUN_FUNC = os.environ["RUN_FUNC"]
    print(f"{RUN_FUNC=}")
    eval(RUN_FUNC)()