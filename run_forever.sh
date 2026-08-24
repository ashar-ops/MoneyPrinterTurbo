#!/bin/bash
export PATH="/home/asharcodes/.local/bin:$PATH"
cd ~/MoneyPrinterTurbo

while true; do
  ./run_one.sh
  sleep 30
done
