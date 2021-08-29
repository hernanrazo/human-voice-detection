#!/bin/sh

echo "Creating data splits, annotations, and transforms for all .wav files ..."
python -c 'from utils.ffnn_utils import prepare_dataset; prepare_dataset()'
echo "done"
