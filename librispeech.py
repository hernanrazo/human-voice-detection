import os
import glob
import shutil
'''
Moves all .flac files from the Librispeech dataset from its 
original file structure to one giant subdir. This ignores
the other .txt files and labels that come with the original
dataset.

Ignore this if you are not using the LibriSpeech dataset or 
if you end up using the original file structure.
'''

def main():

    root = str(os.getcwd()) + '/data/LibriSpeech/train-clean-100/'
    dest = str(os.getcwd()) + '/data/voice/flac/'
    
    for subdirs, dirs, files in os.walk(root):
        for file in files:
            if file.endswith('.flac'):
                print('Moving ' + str(file))
                shutil.move(str(subdirs + os.sep + file), dest)

if __name__ == '__main__':
    main()
