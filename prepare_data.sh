#!/bin/sh

echo "Creating data splits, annotations, and transforms for all .wav files ..."
python -c 'import utils; utils.prepare_dataset()'
echo "done"
