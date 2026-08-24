#!/bin/bash
PEXELS_KEY="lMR8ydaHBjLaKvAzH0LaRSiitPSYH1GaD5WJIvnXoBDnF6miHnEPN6g3"

RESPONSE=$(curl -s -D - -o /dev/null \
  -H "Authorization: $PEXELS_KEY" \
  "https://api.pexels.com/v1/search?query=nature&per_page=1")

REMAINING=$(echo "$RESPONSE" | grep -i "X-Ratelimit-Remaining" | tr -d '\r' | awk '{print $2}')
RESET=$(echo "$RESPONSE" | grep -i "X-Ratelimit-Reset" | tr -d '\r' | awk '{print $2}')

echo "Pexels remaining: $REMAINING (resets at $(date -d @$RESET))"

if [ "$REMAINING" -lt 20 ]; then
  echo "Quota too low — skipping this run"
  exit 1
fi
