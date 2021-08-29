import os
import shutil
import warnings
import argparse
import librosa
import torch
import torchvision
from PIL import Image
from ffnn.model import FFNN
from cnn.model import CNN
from utils.gen_utils import create_dir
from utils.ffnn_utils import apply_transforms, transforms_to_tensor
from utils.cnn_utils import get_melss

warnings.filterwarnings('ignore', category=UserWarning)

def main():

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # other
    root_dir = str(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
    to_tensor = torchvision.transforms.ToTensor()
    
    # get model path argument
    parser = argparse.ArgumentParser()
    parser.add_argument('ffnn_path', type=str, help='Path to feed forward neural network')
    parser.add_argument('cnn_path', type=str, help='Path to convolutional neural network')
    args = parser.parse_args()

    # get models in eval mode
    ffnn = FFNN()
    ffnn_path = os.path.abspath('saved_models/' + args.ffnn_path)
    ffnn.load_state_dict(torch.load(ffnn_path), strict=False)
    ffnn = ffnn.to(device)
    ffnn.eval()

    cnn = CNN()
    cnn_path = os.path.abspath('saved_models/' + args.cnn_path)
    cnn.load_state_dict(torch.load(cnn_path), strict=False)
    cnn = cnn.to(device)
    cnn.eval()

    # create temp dir to save melss image for current inference
    create_dir('temp')
    
    # get transforms and spectrogram image
    transforms = apply_transforms('data/voice/wav/7447-91187-0034.wav')
    melss = get_melss('data/voice/wav/7447-91187-0034.wav', 'temp/test.jpg')

    # convert transforms dict to tensor and 
    # apply transforms to melss image
    transforms = transforms_to_tensor(transforms)
    melss = Image.open('temp/test.jpg')
    melss = melss.resize((32, 32))
    melss = to_tensor(melss)
    melss = melss.to(device)
    
    # make predictions
    ffnn_pred = ffnn(transforms)
    cnn_pred = cnn(melss.unsqueeze(0))

    # if both models agree that the audio is a voice, return voice
    # else, return not_voice
    if ffnn_pred[1] > 0.85 and cnn_pred[0][1] > 0.85:
        print(ffnn_pred)
        print(cnn_pred)
        print('\nvoice\n')
    else:
        print(ffnn_pred)
        print(cnn_pred)
        print('\nnot_voice\n')

    # delete temp dir after completion
    if os.path.isdir(root_dir + '/voice_detect/temp'):
        print('\ndeleting temp dir ...\n')
        os.remove(root_dir + '/voice_detect/temp/test.jpg')
        shutil.rmtree(root_dir + '/voice_detect/temp')
    else:
        print('temp dir does not exist...\n')

    print('Inference complete ...')
if __name__ == '__main__':
    main()
