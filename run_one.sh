#!/bin/bash
export PATH="/home/asharcodes/.local/bin:$PATH"
cd ~/MoneyPrinterTurbo

LOCK=/tmp/mpt.lock
DRIVE_FOLDER_ID="18e8n-JQG4YArg3OtMC0gWaT0E3ppztAr"
exec 200>$LOCK
flock -n 200 || { echo "Already running, skipping"; exit 1; }

./check_pexels_quota.sh || exit 1

TOPIC=$(head -n 1 topics.txt)
if [ -z "$TOPIC" ]; then
    echo "Topic queue empty"
    exit 0
fi
tail -n +2 topics.txt > topics.tmp && mv topics.tmp topics.txt

PROMPT="You are a viral YouTube Shorts storyteller. Write an engaging script about this topic. Start with a strong hook that grabs attention immediately. Use simple conversational English. Make it punchy and surprising. Keep it around 130 words."

MAX_RETRIES=3
ATTEMPT=0
SUCCESS=0

echo "Processing topic: $TOPIC"

# Run CLI with Retry Logic for Gemini empty responses
until [ $SUCCESS -eq 1 ]; do
    /home/asharcodes/.local/bin/uv run python cli.py \
      --video-subject "$TOPIC" \
      --voice-name "deepgram:aura-2-zeus-en" \
      --video-script-prompt "$PROMPT" \
      --video-clip-duration 2 \
      --font-name "Anton.ttf" \
      --font-size 60 \
      --text-fore-color "#FFFFFF" \
      --stroke-color "#000000" \
      --stroke-width 4.5
    
    if [ $? -eq 0 ]; then
        SUCCESS=1
    else
        ATTEMPT=$((ATTEMPT+1))
        if [ $ATTEMPT -ge $MAX_RETRIES ]; then
            echo "Failed after $MAX_RETRIES attempts, skipping topic: $TOPIC"
            break
        fi
        echo "Gemini returned empty response — retrying in 30s (attempt $((ATTEMPT+1))/$MAX_RETRIES)..."
        sleep 30
    fi
done

# If successful, upload to Google Drive and clean up
if [ $SUCCESS -eq 1 ]; then
    LATEST=$(find ./storage/tasks -name "final-*.mp4" -newermt "-15 minutes" | sort | tail -n 1)
    if [ -n "$LATEST" ]; then
        /home/asharcodes/.local/bin/uv run python3 gdrive_upload.py "$LATEST" "$DRIVE_FOLDER_ID"
        if [ $? -eq 0 ]; then
            TASKDIR=$(dirname "$LATEST")
            rm -rf "$TASKDIR"
            echo "Uploaded and cleaned up: $TASKDIR"
        else
            echo "Upload failed — keeping local files for $LATEST"
        fi
    fi
fi
