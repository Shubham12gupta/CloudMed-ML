#!/bin/bash

echo "Downloading consultation models..."

mkdir -p /app/models

aws s3 sync \
s3://mlmodel2026/Medical-ml-v2/models \
/app/models

echo "Models downloaded"

ls -lah /app/models

gunicorn -w 2 -b 0.0.0.0:5000 app:app
