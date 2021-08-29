#!/bin/sh

echo "Creating data splits and melss .jpg files for all .wav files ..."
python -c 'from utils.cnn_utils import prepare_dataset; prepare_dataset()'
echo "done"
