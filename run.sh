DATA="icdm22"   # Choose in {acl18, cikm18, icdm22}.
GPU=0           # GPU id
FILES_PATH="files"
NOISE=0.0
# Prepare dataset
if [ ! -f "data.zip" ]; then
    echo "Please download data.zip first"
    exit 1
fi
unzip -n "data.zip"   # -n: don't overwrite existing files

# Change dir to src
cd src

# Run sentencepiece to tokenize tweets (only once is enough)
python embed.py --data "$DATA"

# ---- Loop Seeds ----
for SEED in $(seq 0 7); do
    
    # ---- Loop Noise Ratios (Experiment Q2) ----
    # We test 0.0 (clean), 0.2, 0.4, 0.6, 0.8
    # for NOISE in 0.0 0.2 0.4 0.6 0.8; do
    
    echo "=================================================="
    echo " PROCESSING: SEED=$SEED | NOISE= "
    echo "=================================================="
    
    # Create a unique output directory for this specific Noise level
    # OUT="out/${DATA}-${SEED}-${NOISE}"
    OUT="out/${DATA}-${SEED}"
    mkdir -p "$OUT"

    # 1) self-supervised pre-training (tweets)
    # Note: We re-run this to ensure 'tweet_model.pth' exists in the new $OUT folder
    echo "Starting tweet pretraining..."
    python pretrain.py \
        --data "$DATA" \
        --gpu "$GPU" \
        --seed "$SEED" \
        --out "$OUT" > /dev/null  # hiding logs to keep terminal clean

    # 2) self-supervised pre-training (news)
    # News model is usually saved globally, but running it ensures dependencies are met
    echo "Starting news pretraining..."
    python pretrain_news.py \
        --files-path "$FILES_PATH" \
        --data "$DATA" \
        --gpu "$GPU" \
        --seed "$SEED" \
        --out "$OUT" > /dev/null

    # 3) preprocessing (feature generation) - THIS IS WHERE NOISE IS APPLIED
    # echo "Started preprocessing with Noise Ratio: $NOISE"
    python preprocess.py \
        --data "$DATA" \
        --gpu "$GPU" \
        --seed "$SEED" \
        --out "$OUT" \
        --global-trend false \
        --local-trend false \
        --reliable-tweet-trend true \
        --global-news-trend true \
        --window 10 \
        # --noise-ratio "$NOISE"  # <--- PASSING THE NOISE RATIO HERE

    # 4) train ALSTM (DAMA Model)
    echo "Started training ALSTM..."
    python predict.py \
        --data "$DATA" \
        --gpu "$GPU" \
        --seed "$SEED" \
        --out "$OUT" \
        --hidden-dim 64 \
        --l2-norm 1.0 \
        --lr 1e-3

        echo "Finished Seed $SEED with Noise $NOISE"
    # done
done

echo "ALL EXPERIMENTS DONE."