#!/bin/bash
# run_tuning.sh

DATA="icdm22"   # Choose in {acl18, cikm18, icdm22}
GPU=0           # GPU id
FILES_PATH="files"
NOISE=0.0

# Prepare dataset
if [ ! -f "data.zip" ]; then
    echo "Please download data.zip first"
    exit 1
fi
unzip -n "data.zip"

cd src

# Run sentencepiece to tokenize tweets (only once is enough)
python embed.py --data "$DATA"

for SEED in $(seq 0 4); do
    echo "=================================================="
    echo " PRETRAINING FOR SEED $SEED "
    echo "=================================================="
    
    # Create a base directory to store the pretrained tweet model for this seed
    BASE_OUT="out/${DATA}-seed${SEED}-base"
    mkdir -p "$BASE_OUT"

    # 1. Pretrain Tweets (Saves tweet_model.pth in BASE_OUT)
    echo "Starting tweet pretraining..."
    python pretrain.py --data "$DATA" --gpu "$GPU" --seed "$SEED" --out "$BASE_OUT" > /dev/null
    
    # 2. Pretrain News (Saves globally to out/${DATA}-newsmodel/pretrain_news_model.pt)
    echo "Starting news pretraining..."
    python pretrain_news.py --files-path "$FILES_PATH" --data "$DATA" --gpu "$GPU" --seed "$SEED" --out "$BASE_OUT" > /dev/null

    echo "=== 1. Tuning Window Size (Seed $SEED) ==="
    for WINDOW in 5 10 15 20; do
        WIN_OUT="out/${DATA}-win${WINDOW}-seed${SEED}"
        mkdir -p "$WIN_OUT"
        
        # CRUCIAL: Copy the pretrained tweet model so preprocess.py can find it!
        cp "${BASE_OUT}/tweet_model.pth" "${WIN_OUT}/tweet_model.pth"
        
        # 3. Preprocess
        echo "Preprocessing Window $WINDOW..."
        python preprocess.py \
            --data "$DATA" \
            --gpu "$GPU" \
            --seed "$SEED" \
            --out "$WIN_OUT" \
            --global-trend false \
            --local-trend false \
            --reliable-tweet-trend true \
            --global-news-trend true \
            --window "$WINDOW" \
            --noise-ratio "$NOISE"
        
        # 4. Predict
        echo "Training ALSTM (Window $WINDOW)..."
        python predict.py \
            --data "$DATA" \
            --gpu "$GPU" \
            --seed "$SEED" \
            --out "$WIN_OUT" \
            --hidden-dim 32 \
            --l2-norm 1.0 \
            --lr 1e-3
    done

    echo "=== 2. Tuning Hidden Dimension (Seed $SEED) ==="
    # We use Window 10 as our baseline to test hidden dimensions
    for HDIM in 16 32 64 128; do
        HDIM_OUT="out/${DATA}-hdim${HDIM}-seed${SEED}"
        mkdir -p "$HDIM_OUT"
        
        # CRUCIAL: We don't need to re-preprocess! Just copy the features.pkl from the Window 10 run.
        WIN10_OUT="out/${DATA}-win10-seed${SEED}"
        cp "${WIN10_OUT}/features.pkl" "${HDIM_OUT}/features.pkl"
        
        # Predict
        echo "Training ALSTM (Hidden Dim $HDIM)..."
        python predict.py \
            --data "$DATA" \
            --gpu "$GPU" \
            --seed "$SEED" \
            --out "$HDIM_OUT" \
            --hidden-dim "$HDIM" \
            --l2-norm 1.0 \
            --lr 1e-3
    done
done

echo "ALL TUNING EXPERIMENTS DONE."