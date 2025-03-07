#!/bin/bash

export PORT=5000  # Change if needed

echo "Starting Flask app..."
gunicorn -w 4 -b 0.0.0.0:$PORT app:app
