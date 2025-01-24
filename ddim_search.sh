# Run second experiment
CUDA_VISIBLE_DEVICES="2,3" torchrun --nproc_per_node 2 --nnodes 1 train_emage_audio.py \
   --config ./configs/infp_audio.yaml \
   --evaluation \
   --wandb \

# Find the latest output folder and test_100000 cache after experiment
LATEST_OUTPUT=$(ls -td ../outputs/*/ | head -n 1)
CACHE_PATH="${LATEST_OUTPUT}test_100000"
echo $CACHE_PATH
# Run reconstruction with v-prediction type
python ./datasets/reconstruction.py --cache_dir "$CACHE_PATH" --save_dir "/home/weili/haiyang/PantoMatrix/HDTF/test_reconstructions/no_norm_ar/"

git checkout infp-ddim
# Run second experiment
CUDA_VISIBLE_DEVICES="2,3" torchrun --nproc_per_node 2 --nnodes 1 train_emage_audio.py \
   --config ./configs/infp_audio.yaml \
   --evaluation \
   --wandb \

# Find the latest output folder and test_100000 cache after experiment
LATEST_OUTPUT=$(ls -td ../outputs/*/ | head -n 1)
CACHE_PATH="${LATEST_OUTPUT}test_100000"
echo $CACHE_PATH
# Run reconstruction with v-prediction type
python ./datasets/reconstruction.py --cache_dir "$CACHE_PATH" --save_dir "/home/weili/haiyang/PantoMatrix/HDTF/test_reconstructions/no_norm/"

# Run first experiment
CUDA_VISIBLE_DEVICES="2,3" torchrun --nproc_per_node 2 --nnodes 1 train_emage_audio.py \
   --config ./configs/infp_audio_v.yaml \
   --evaluation \
   --wandb \

# Find the latest output folder and test_100000 cache after experiment
LATEST_OUTPUT=$(ls -td ../outputs/*/ | head -n 1)
CACHE_PATH="${LATEST_OUTPUT}test_100000"
echo $CACHE_PATH
# Run reconstruction with sample prediction type
python ./datasets/reconstruction.py --cache_dir "$CACHE_PATH" --save_dir "/home/weili/haiyang/PantoMatrix/HDTF/test_reconstructions/v_no_norm/"

# Run second experiment
CUDA_VISIBLE_DEVICES="2,3" torchrun --nproc_per_node 2 --nnodes 1 train_emage_audio.py \
   --config ./configs/infp_audio_e.yaml \
   --evaluation \
   --wandb \

# Find the latest output folder and test_100000 cache after experiment
LATEST_OUTPUT=$(ls -td ../outputs/*/ | head -n 1)
CACHE_PATH="${LATEST_OUTPUT}test_100000"
echo $CACHE_PATH
# Run reconstruction with v-prediction type
python ./datasets/reconstruction.py --cache_dir "$CACHE_PATH" --save_dir "/home/weili/haiyang/PantoMatrix/HDTF/test_reconstructions/e_no_norm/"