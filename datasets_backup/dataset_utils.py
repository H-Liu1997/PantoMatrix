from pathlib import Path
import joblib
import torch
from einops import repeat, rearrange
import numpy as np

# from .whisper_audio2feature import Audio2Feature
from .wave2vec_wrapper import (
    wav2vec2_wrapper_new,
    feature2chunks
)
from .emo_image import (
    tonp,
)

### AUDIO PROCESSING ###
# 1
def get_audio_enc(
    audio_fea_type,
    device='cpu',
):
    print(f'>>>>>>>>>> LOAD WAV2VEC2')
    if audio_fea_type=='whisper':
        audio_processor = Audio2Feature(model_path="pretrained_weights/tiny.pt",device=device,)
    elif audio_fea_type in [
        'wav2vec2_type2',
        'wav2vec2_type5',
        'wav2vec2_type6',
        'wav2vec2_type7',
    ]:
        audio_processor = wav2vec2_wrapper_new(device=device)
    else:
        raise NotImplementedError
        # audio_processor = wav2vec2_wrapper(device=device)
    print(f'>>>>>>>>>> LOAD DONE')
    return audio_processor

# 2
def get_wav_fea(
    wav_path,
    audio_fea_type,
    audio_processor,
    actual_fps_after_downsample,
    load_w_save=False,
):
    wav_path = str(wav_path)
    # import ipdb;ipdb.set_trace()
    # <========= no common used function, only for temporary usage.
    def tmp_func_get_fea(
        wav_fea_pt_path,
        only_last_features,
    ):
        def run_fea_ex_and_save():
            assert Path(wav_path).exists(), f'{wav_path=} not exists.'
            wav_fea = audio_processor.forward(
                wav_path, 
                fps=actual_fps_after_downsample,
                cvt_to_chunk=False,
                only_last_features=only_last_features,                
            )
            if load_w_save:
                joblib.dump({
                    'wav_fea': tonp(wav_fea),
                }, wav_fea_pt_path)  
            return wav_fea
            
        if Path(wav_fea_pt_path).exists():
            try:
                tmp_data = joblib.load(wav_fea_pt_path)
                wav_fea = torch.from_numpy(tmp_data['wav_fea'])
            except:
                wav_fea = run_fea_ex_and_save()
        else:
            wav_fea = run_fea_ex_and_save()
        return wav_fea
    # ==========>

    if audio_fea_type=='whisper':
        wav_fea = audio_processor.feature2chunks(audio_processor.audio2feat(wav_path),fps=actual_fps_after_downsample)
    elif (
        audio_fea_type=='wav2vec2_type2'
    ):
        wav_fea_pt_path = wav_path.replace('.wav','.wav_fea_jpkl_type2')
        wav_fea = tmp_func_get_fea(wav_fea_pt_path, only_last_features=False)
        wav_fea = feature2chunks(wav_fea,actual_fps_after_downsample,audio_feat_length = [2,2]) # 10/50=0.2s
        assert wav_fea.shape[1]==13*10, f'{wav_fea.shape=} not match.'
    elif (
        audio_fea_type=='wav2vec2_type5'
    ):
        wav_fea_pt_path = wav_path.replace('.wav','.wav_fea_jpkl_type5')
        wav_fea = tmp_func_get_fea(wav_fea_pt_path, only_last_features=True)
        wav_fea = feature2chunks(wav_fea,actual_fps_after_downsample,audio_feat_length = [2,2]) 
        assert wav_fea.shape[1]==1*10, f'{wav_fea.shape=} not match.'
    elif (
        audio_fea_type=='wav2vec2_type6'
    ):
        wav_fea_pt_path = wav_path.replace('.wav','.wav_fea_jpkl_type2')
        wav_fea = tmp_func_get_fea(wav_fea_pt_path, only_last_features=False)
        wav_fea = feature2chunks(wav_fea,actual_fps_after_downsample,audio_feat_length = [1,1]) 
        assert wav_fea.shape[1]==13*5, f'{wav_fea.shape=} not match.'
    elif (
        audio_fea_type=='wav2vec2_type7'
    ):

        # def get_history_stack(fea,m):
        #     # fea: F,768
        #     fea = torch.cat([fea[:m], fea, fea[-m:]], axis=0)
        #     fea = torch.stack([fea[idx-m:idx+m+1] for idx in range(m, fea.shape[0]-m)], dim=0)
        #     return fea
        
        wav_fea_pt_path = wav_path.replace('.wav','.wav_fea_jpkl_type2')
        wav_fea = tmp_func_get_fea(wav_fea_pt_path, only_last_features=False)
        # import ipdb;ipdb.set_trace()
        wav_fea = wav_fea[:,-1:,:]
        wav_fea = feature2chunks(wav_fea,actual_fps_after_downsample,audio_feat_length = [3,3]) #14
        # wav_fea = get_history_stack(wav_fea, 7)
        wav_fea = wav_fea[:,::3,:]
        wav_fea = wav_fea.flatten(1)
    else:
        raise NotImplementedError
    return wav_fea

# 3
def cvt_audio_fea(
    wav2vec_embeds,
    audio_fea_type,
    audio_cond_pos_embed,
):
    if audio_fea_type in [
        'whisper',
        'wav2vec2_type2',
        'wav2vec2_type5',
        'wav2vec2_type6',
    ]:
        wav2vec_embeds = rearrange(wav2vec_embeds, "b f n c -> (b f) n c")
        wav2vec_embeds = audio_cond_pos_embed(wav2vec_embeds)
    elif audio_fea_type == 'wav2vec2_type7':
        # import ipdb;ipdb.set_trace()
        wav2vec_embeds = torch.cat(
            [
                wav2vec_embeds:= rearrange(wav2vec_embeds, "b f c -> (b f) 1 c"),
                torch.zeros_like(wav2vec_embeds)
            ],
        dim=1)
    else:
        raise NotImplementedError
    return wav2vec_embeds 

### DATA LOADING ###
def parse_vid_id_from_path(vid_path):
    vid_id = (
        str(Path(vid_path).stem)
        .replace('_25fps.mp4','')
        .replace('_origin.mp4','')
        .replace('.mp4','')
        .replace('_25fps','')
        .replace('_origin','')
    )
    return vid_id

def parse_protocol2(data_info):
    data_name_meta = []
    mp4_suffix = data_info.mp4_suffix
    wav_suffix = data_info.wav_suffix
    meta_pt = data_info.meta_pt
    root = data_info.root
    data_meta = torch.load(meta_pt)
    for meta in data_meta:
        vid_id = meta['vid_id']
        vid_id = parse_vid_id_from_path(vid_id)
        vid_id = vid_id.split('/')[-1]
        # filter error id
        if vid_id in []:
            continue
        data_name_meta.append({
            'video_path': str(Path(root,f'{vid_id}{mp4_suffix}.mp4')),
            'wav_path': str(Path(root,f'{vid_id}{wav_suffix}.wav')),
            'bbox': meta['bbox'],
        })   
    return data_name_meta  

### FPS UTILS ###
def convert_var_to_nay_fps(var, ori_var_fps, target_fps):
    """
    Args:
        var: (T,...)
        ori_var_fps: 25
    """
    original_frame_count = len(var)
    target_frame_count = int((original_frame_count / ori_var_fps) * target_fps)
    target_frame_indices = [
        min(
            round((target_fps_pos / target_fps) * ori_var_fps),
            original_frame_count - 1,
        )
        for target_fps_pos in range(target_frame_count)
    ]
    target_frame_indices = np.array(target_frame_indices)
    return var[target_frame_indices]

### SAMPLING UTILS ###
def get_dataset_sample_rate(dset_sample_rate, target_rate):
        if target_rate is None:
            return dset_sample_rate
        target_rate = np.array(target_rate)
        
        dset_sample_rate_new = (dset_sample_rate * target_rate)
        dset_sample_rate_new = dset_sample_rate_new / np.sum(dset_sample_rate_new)
        return dset_sample_rate_new

def crop_with_padding(image, center, size=512):
    """
    Crop a region of a specified size from the given center point, 
    filling the area outside the image boundary with zeros.
    
    :param image: The input image in NumPy array form, shape (H, W, C)
    :param center: The center point (y, x) to start cropping from
    :param size: The size of the cropped region (default is 512)
    :return: The cropped region with padding, shape (size, size, C)
    """
    h, w = image.shape[:2]  # Get the height and width of the image
    half_size = size // 2  # Half the size for the cropping region

    # Calculate the top-left and bottom-right coordinates of the cropping region
    y1 = max(center[0] - half_size, 0)  # Ensure the y1 index is not less than 0
    x1 = max(center[1] - half_size, 0)  # Ensure the x1 index is not less than 0
    y2 = min(center[0] + half_size, h)  # Ensure the y2 index does not exceed the image height
    x2 = min(center[1] + half_size, w)  # Ensure the x2 index does not exceed the image width

    # Create a zero-filled array for padding
    cropped = np.zeros((size, size, image.shape[2]), dtype=image.dtype)
    
    # Copy the valid region from the original image to the cropped region
    cropped[(y1 - (center[0] - half_size)):(y2 - (center[0] - half_size)),
            (x1 - (center[1] - half_size)):(x2 - (center[1] - half_size))] = image[y1:y2, x1:x2]
    
    return cropped

def get_metadata_path(video_path):
    base_dir, filename = video_path.rsplit("/", 1)
        
    if "videos_resampled" in base_dir:
        if "+" in filename:
            scene_identifier = filename.split("+")[0]
        else:
            scene_identifier = filename.split(".")[0]
        metadata_dir = base_dir.replace("/videos_resampled", "/metadata")
    else:
        scene_identifier = filename.split(".")[0]
        metadata_dir = base_dir.replace("/videos", "/metadata")
    
    metadata_path = f"{metadata_dir}/{scene_identifier}"
    metadata_file = Path(metadata_path) / "metadata.npz"

    return metadata_file

def generate_crop_bounding_box(h, w, center, size=512):
    """
    Crop a region of a specified size from the given center point, 
    filling the area outside the image boundary with zeros.
    
    :param image: The input image in NumPy array form, shape (H, W, C)
    :param center: The center point (y, x) to start cropping from
    :param size: The size of the cropped region (default is 512)
    :return: The cropped region with padding, shape (size, size, C)
    """
    half_size = size // 2  # Half the size for the cropping region

    # Calculate the top-left and bottom-right coordinates of the cropping region
    y1 = max(center[0] - half_size, 0)  # Ensure the y1 index is not less than 0
    x1 = max(center[1] - half_size, 0)  # Ensure the x1 index is not less than 0
    y2 = min(center[0] + half_size, h)  # Ensure the y2 index does not exceed the image height
    x2 = min(center[1] + half_size, w)  # Ensure the x2 index does not exceed the image width
    return [x1, y1, x2, y2]

def crop_from_bbox(image, center, bbox, size=512):
    """
    Crop a region of a specified size from the given center point, 
    filling the area outside the image boundary with zeros.
    
    :param image: The input image in NumPy array form, shape (H, W, C)
    :param center: The center point (y, x) to start cropping from
    :param size: The size of the cropped region (default is 512)
    :return: The cropped region with padding, shape (size, size, C)
    """
    h, w = image.shape[:2]  # Get the height and width of the image
    x1, y1, x2, y2 = bbox
    half_size = size // 2  # Half the size for the cropping region
    # Create a zero-filled array for padding
    cropped = np.zeros((size, size, image.shape[2]), dtype=image.dtype)
    
    # Copy the valid region from the original image to the cropped region
    cropped[(y1 - (center[0] - half_size)):(y2 - (center[0] - half_size)),
            (x1 - (center[1] - half_size)):(x2 - (center[1] - half_size))] = image[y1:y2, x1:x2]
    
    return cropped