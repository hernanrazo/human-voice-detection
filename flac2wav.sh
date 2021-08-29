#!/bin/sh

echo "Converting .flac files to .wav"
python -c 'from utils.gen_utils import flac2wav; flac2wav()'
