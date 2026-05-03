echo "================================================================"
echo "Job started on $(hostname) at $(date)"
echo "================================================================"

module purge
module load python/3.9.12
module load cuda/11.8.0
module load ffmpeg
source ~/.myenv/bin/activate

echo "Python : $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA   : $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPUs   : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "----------------------------------------------------------------"

export MASTER_PORT=$((29500 + ${SLURM_JOB_ID:-$$} % 1000))
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

#----------------------------------------------------------------#
#  DATA PATHS
#----------------------------------------------------------------#
TRAIN_ROOT=/data/Data/ASVSpoofData_2019/train/LA/ASVspoof2019_LA_train/flac
TRAIN_PROTOCOL=/data/Data/ASVSpoofData_2019/train/LA/ASVspoof2019_train_protocol_with_speaker.txt
DEV_ROOT=/data/Data/ASVSpoofData_2019/train/LA/ASVspoof2019_LA_dev/flac
DEV_PROTOCOL=/data/Data/ASVSpoofData_2019/train/LA/ASVspoof2019_dev_protocol_with_speaker.txt

#----------------------------------------------------------------#
#  EXPERIMENT VARIABLES  — edit here
#----------------------------------------------------------------#
EXP_NAME=aug_ntxent_v1

MODEL_PATH=facebook/wav2vec2-xls-r-300m
MODEL=wav2vec2-xls-r-300m

# NT-Xent / augmentation
N_VIEWS=5
TEMPERATURE=0.2
AUG_MODE=chain

# Training
SEED=1337
BATCH_SIZE=32
EPOCHS=100
ENC_LR=1e-5
HEAD_LR=5e-4
WEIGHT_DECAY=3e-3
MAX_DURATION_SECONDS=10
FREEZE_ENCODER=0        # 0 = fine-tune (default)
TRAIN_BONAFIDE_ONLY=1   # 1 = bonafide-only (recommended: faster + NT-Xent has no use for spoof samples)
ACCUM_STEPS=1
NUM_WORKERS=4           # DataLoader workers per process (increase on fast NFS)

# Noise / RIR augmentation directories
NOISE_DIR=$(dirname "$0")/bg_noise
RIR_DIR=""              # leave empty if no RIR files

# Plots
DR_METHOD=umap          # umap | tsne

# Stage 2 classifier
S2_BATCH=64
S2_EPOCHS=200
S2_LR=1e-4
S2_WD=1e-4
S2_PATIENCE=15
S2_HIDDEN=64
S2_DROPOUT=0.2

# Evaluation datasets (comma-separated)
DATASET_LIST="itw,asv19,fakexpose"

# FamousFigures speaker filter (leave empty for all)
FF_SPEAKERS=""

#----------------------------------------------------------------#
#  DERIVED PATHS
#----------------------------------------------------------------#
cd "$(dirname "$0")" || exit 1

MODEL_TAG=${MODEL_PATH//\//__}

CKPT_ROOT=checkpoints_stage1/${EXP_NAME}/${MODEL_TAG}
STAGE1_CKPT=${CKPT_ROOT}/${MODEL_TAG}_ntxent_best.pt

STAGE2_DIR=checkpoints_stage2/${EXP_NAME}
STAGE2_CKPT=${STAGE2_DIR}/stage2_mlp_best.pt

SCORES_DIR=scores

#----------------------------------------------------------------#
#  GPUs to use (comma-separated list, e.g. "0,1" or "2")
#----------------------------------------------------------------#
GPU_IDS=2
NUM_GPUS=$(echo "${GPU_IDS}" | tr ',' '\n' | wc -l | tr -d ' ')
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
echo "Using GPU_IDS=${GPU_IDS}  NUM_GPUS=${NUM_GPUS}"

LAUNCHER="torchrun --nproc_per_node=${NUM_GPUS} --rdzv_endpoint=localhost:${MASTER_PORT}"

echo "================================================================"
echo " EXPERIMENT : ${EXP_NAME}"
echo " STAGE 1 ckpt: ${STAGE1_CKPT}"
echo " STAGE 2 ckpt: ${STAGE2_CKPT}"
echo " Datasets  : ${DATASET_LIST}"
echo "================================================================"

mkdir -p logs

#----------------------------------------------------------------#
#  STAGE 1 — NT-Xent training
#----------------------------------------------------------------#
echo ""
echo ">>> [Stage 1] NT-Xent training  $(date)"

${LAUNCHER} train_aug_ntxent.py \
    --model_name      "${MODEL_PATH}" \
    --train_root      "${TRAIN_ROOT}" \
    --train_protocol  "${TRAIN_PROTOCOL}" \
    --dev_root        "${DEV_ROOT}" \
    --dev_protocol    "${DEV_PROTOCOL}" \
    --save_dir        "${CKPT_ROOT}" \
    --epochs     "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --n_views    "${N_VIEWS}" \
    --temperature "${TEMPERATURE}" \
    --aug_mode   "${AUG_MODE}" \
    --noise_dir  "${NOISE_DIR}" \
    --rir_dir    "${RIR_DIR}" \
    --enc_lr     "${ENC_LR}" \
    --head_lr    "${HEAD_LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --max_duration_seconds "${MAX_DURATION_SECONDS}" \
    --freeze_encoder "${FREEZE_ENCODER}" \
    --train_bonafide_only "${TRAIN_BONAFIDE_ONLY}" \
    --accum_steps "${ACCUM_STEPS}" \
    --seed        "${SEED}" \
    --num_workers "${NUM_WORKERS}"

if [ ! -f "${STAGE1_CKPT}" ]; then
    echo "[ERROR] Stage 1 checkpoint not found: ${STAGE1_CKPT}"
    exit 1
fi
echo ">>> [Stage 1] Done  $(date)"

#----------------------------------------------------------------#
#  PLOTS — UMAP / t-SNE embeddings (ASV19 + ITW)
#----------------------------------------------------------------#
echo ""
echo ">>> [Plots] Visualising embeddings  $(date)"

python plot_embeddings.py \
    --dataset    both \
    --ckpt_path  "${STAGE1_CKPT}" \
    --model_name "${MODEL_PATH}" \
    --dr_method  "${DR_METHOD}" \
    --exp_name   "${EXP_NAME}" \
    --plots_dir  "plots/${EXP_NAME}" \
    --max_duration_seconds "${MAX_DURATION_SECONDS}" \
    --batch_size 64 --num_workers "${NUM_WORKERS}"

echo ">>> [Plots] Done  $(date)"

#----------------------------------------------------------------#
#  STAGE 2 — MLP classifier
#----------------------------------------------------------------#
echo ""
echo ">>> [Stage 2] MLP classifier training  $(date)"

python train_stage2_mlp.py \
    --stage1_ckpt     "${STAGE1_CKPT}" \
    --model_name      "${MODEL_PATH}" \
    --train_root      "${TRAIN_ROOT}" \
    --train_protocol  "${TRAIN_PROTOCOL}" \
    --dev_root        "${DEV_ROOT}" \
    --dev_protocol    "${DEV_PROTOCOL}" \
    --save_dir    "${STAGE2_DIR}" \
    --batch_size  "${S2_BATCH}" \
    --epochs      "${S2_EPOCHS}" \
    --lr          "${S2_LR}" \
    --weight_decay "${S2_WD}" \
    --patience    "${S2_PATIENCE}" \
    --hidden_dim  "${S2_HIDDEN}" \
    --dropout     "${S2_DROPOUT}" \
    --max_duration_seconds "${MAX_DURATION_SECONDS}" \
    --num_workers "${NUM_WORKERS}" --seed "${SEED}"

if [ ! -f "${STAGE2_CKPT}" ]; then
    echo "[ERROR] Stage 2 checkpoint not found: ${STAGE2_CKPT}"
    exit 1
fi
echo ">>> [Stage 2] Done  $(date)"

#----------------------------------------------------------------#
#  EVALUATION — score datasets + print EER
#----------------------------------------------------------------#
echo ""
echo ">>> [Eval] Scoring datasets: ${DATASET_LIST}  $(date)"

python eval_datasets.py \
    --exp_name    "${EXP_NAME}" \
    --datasets    "${DATASET_LIST}" \
    --model_name  "${MODEL_PATH}" \
    --stage1_ckpt "${STAGE1_CKPT}" \
    --stage2_ckpt "${STAGE2_CKPT}" \
    --scores_dir  "${SCORES_DIR}" \
    --batch_size  32 \
    --num_workers "${NUM_WORKERS}" \
    --max_duration_seconds "${MAX_DURATION_SECONDS}" \
    --ff_speakers "${FF_SPEAKERS}" \
    --print_eer

echo ""
echo "================================================================"
echo "Pipeline finished at $(date)"
echo "================================================================"
