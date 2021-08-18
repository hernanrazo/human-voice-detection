#!/bin/sh

echo "Converting .flac files to .wav"
python -c 'import utils; utils.flac2wav()'
