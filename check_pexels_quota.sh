#!/bin/bash

# 1. Securely read the Pexels API key from config.toml (NO MORE HARDCODED SECRETS)
PEXELS_KEY=$(grep 'pexels_api_keys' config.toml | grep -oE '"[^"]+"' | tr -d '"' | head -n 1)

if [ -z "$PEXELS_KEY" ]; then
    echo "Error: Could not find Pexels API key in config.toml"
    exit 1
fi

# 2. Make the API request and dump headers to a secure temp file
HEADERS_FILE=$(mktemp)
curl -s -D "$HEADERS_FILE" -o /dev/null \
  -H "Authorization: $PEXELS_KEY" \
  "https://api.pexels.com/v1/search?query=nature&per_page=1"

# 3. Parse the headers safely
REMAINING=$(grep -i "X-Ratelimit-Remaining" "$HEADERS_FILE" | tr -d '\r' | awk '{print $2}')
RESET=$(grep -i "X-Ratelimit-Reset" "$HEADERS_FILE" | tr -d '\r' | awk '{print $2}')
LIMIT=$(grep -i "X-Ratelimit-Limit" "$HEADERS_FILE" | tr -d '\r' | awk '{print $2}')

rm -f "$HEADERS_FILE"

# 4. Validate that we actually got numbers back (prevents pipeline crashes)
if [ -z "$REMAINING" ] || ! [[ "$REMAINING" =~ ^[0-9]+$ ]]; then
    echo "Warning: Failed to parse Pexels rate limit headers. Assuming quota is OK to prevent pipeline stall."
    exit 0
fi

# 5. Print status and check threshold
if [ -n "$RESET" ] && [[ "$RESET" =~ ^[0-9]+$ ]]; then
    RESET_DATE=$(date -d @"$RESET" 2>/dev/null || echo "Unknown")
    echo "Pexels API Quota: $REMAINING / $LIMIT remaining (resets at $RESET_DATE)"
else
    echo "Pexels API Quota: $REMAINING / $LIMIT remaining"
fi

if [ "$REMAINING" -lt 20 ]; then
  echo "Quota too low (< 20) — skipping this run"
  exit 1
fi

exit 0
