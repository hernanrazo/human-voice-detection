import os
import csv
import threading
import numpy as np
import shutil
from pydub import AudioSegment
import librosa
import torch
from utils.gen_utils import create_splits

root_dir = str(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
np.set_printoptions(suppress=True)


# assign labels to wav files
def get_label(file_path):
    if 'not_voice' in file_path:
        return 'not_voice'
    else:
        return 'voice'


 # apply transforms needed to prepare data
def apply_transforms(wav_file):
    
    # convert wav file to floating pont time series and get
    # default sample rate (22050)
    x, sr = librosa.load(wav_file, res_type='kaiser_fast')
    
    # get mel-frequency cepstral coefficients
    mfccs = np.mean(librosa.feature.mfcc(y=x, sr=sr, n_mfcc=40).T, axis=0)
    
    #get short-time fourier transform
    stft = np.abs(librosa.stft(x))
    
    # get chromagram
    chromagram = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)

    # get mel_scaled spectrogram
    melss = np.mean(librosa.feature.melspectrogram(x, sr=sr).T, axis=0)

    # get spectral contrast
    spec_contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T, axis=0)
    
    # get tonnetz
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(x), sr=sr).T, axis=0)
    
    return {'mfccs':mfccs, 'chromagram':chromagram, 'melss':melss, 'spec_contrast':spec_contrast, 'tonnetz':tonnetz}


# convert transforms dictionary to a tensor
def transforms_to_tensor(transforms):
    transforms_list = [transforms['mfccs'][0], transforms['mfccs'][1],
                       transforms['mfccs'][2], transforms['mfccs'][3],
                       transforms['mfccs'][4], transforms['mfccs'][5],
                       transforms['mfccs'][6], transforms['mfccs'][7],
                       transforms['mfccs'][8], transforms['mfccs'][9],
                       transforms['mfccs'][10], transforms['mfccs'][11],
                       transforms['mfccs'][12], transforms['mfccs'][13],
                       transforms['mfccs'][14], transforms['mfccs'][15],
                       transforms['mfccs'][16], transforms['mfccs'][17],
                       transforms['mfccs'][18], transforms['mfccs'][19],
                       transforms['mfccs'][20], transforms['mfccs'][21],
                       transforms['mfccs'][22], transforms['mfccs'][23],
                       transforms['mfccs'][24], transforms['mfccs'][25],
                       transforms['mfccs'][26], transforms['mfccs'][27],
                       transforms['mfccs'][28], transforms['mfccs'][29],
                       transforms['mfccs'][30], transforms['mfccs'][31],
                       transforms['mfccs'][32], transforms['mfccs'][33],
                       transforms['mfccs'][34], transforms['mfccs'][35],
                       transforms['mfccs'][36], transforms['mfccs'][37],
                       transforms['mfccs'][38], transforms['mfccs'][39],
                       transforms['chromagram'][0],
                       transforms['chromagram'][1],
                       transforms['chromagram'][2],
                       transforms['chromagram'][3],
                       transforms['chromagram'][4],
                       transforms['chromagram'][5],
                       transforms['chromagram'][6],
                       transforms['chromagram'][7],
                       transforms['chromagram'][8],
                       transforms['chromagram'][9],
                       transforms['chromagram'][10],
                       transforms['chromagram'][11],
                       transforms['melss'][0], transforms['melss'][1], transforms['melss'][2],
                       transforms['melss'][3], transforms['melss'][4], transforms['melss'][5],
                       transforms['melss'][6], transforms['melss'][7], transforms['melss'][8],
                       transforms['melss'][9], transforms['melss'][10], transforms['melss'][11],
                       transforms['melss'][12], transforms['melss'][13], transforms['melss'][14],
                       transforms['melss'][15], transforms['melss'][16], transforms['melss'][17],
                       transforms['melss'][18], transforms['melss'][19], transforms['melss'][20],
                       transforms['melss'][21], transforms['melss'][22], transforms['melss'][23],
                       transforms['melss'][24], transforms['melss'][25], transforms['melss'][26],
                       transforms['melss'][27], transforms['melss'][28], transforms['melss'][29],
                       transforms['melss'][30], transforms['melss'][31], transforms['melss'][32],
                       transforms['melss'][33], transforms['melss'][34], transforms['melss'][35],
                       transforms['melss'][36], transforms['melss'][37], transforms['melss'][38],
                       transforms['melss'][39], transforms['melss'][40], transforms['melss'][41],
                       transforms['melss'][42], transforms['melss'][43], transforms['melss'][44],
                       transforms['melss'][45], transforms['melss'][46], transforms['melss'][47],
                       transforms['melss'][48], transforms['melss'][49], transforms['melss'][50],
                       transforms['melss'][51], transforms['melss'][52], transforms['melss'][53],
                       transforms['melss'][54], transforms['melss'][55], transforms['melss'][56],
                       transforms['melss'][57], transforms['melss'][58], transforms['melss'][59],
                       transforms['melss'][60], transforms['melss'][61], transforms['melss'][62],
                       transforms['melss'][63], transforms['melss'][64], transforms['melss'][65],
                       transforms['melss'][66], transforms['melss'][67], transforms['melss'][68],
                       transforms['melss'][69], transforms['melss'][70], transforms['melss'][71],
                       transforms['melss'][72], transforms['melss'][73], transforms['melss'][74],
                       transforms['melss'][75], transforms['melss'][75], transforms['melss'][76],
                       transforms['melss'][77], transforms['melss'][78], transforms['melss'][79],
                       transforms['melss'][80], transforms['melss'][81], transforms['melss'][82],
                       transforms['melss'][83], transforms['melss'][84], transforms['melss'][85],
                       transforms['melss'][86], transforms['melss'][87], transforms['melss'][88],
                       transforms['melss'][89], transforms['melss'][90], transforms['melss'][91],
                       transforms['melss'][92], transforms['melss'][93], transforms['melss'][94],
                       transforms['melss'][95], transforms['melss'][96], transforms['melss'][97],
                       transforms['melss'][98], transforms['melss'][99], transforms['melss'][100],
                       transforms['melss'][101], transforms['melss'][102], transforms['melss'][103],
                       transforms['melss'][104], transforms['melss'][105], transforms['melss'][106],
                       transforms['melss'][107], transforms['melss'][108], transforms['melss'][109],
                       transforms['melss'][110], transforms['melss'][111], transforms['melss'][112],
                       transforms['melss'][113], transforms['melss'][114], transforms['melss'][115],
                       transforms['melss'][116], transforms['melss'][117], transforms['melss'][118],
                       transforms['melss'][119], transforms['melss'][120], transforms['melss'][121],
                       transforms['melss'][122], transforms['melss'][123], transforms['melss'][124],
                       transforms['melss'][125], transforms['melss'][126], transforms['melss'][127],
                       transforms['spec_contrast'][0],
                       transforms['spec_contrast'][1],
                       transforms['spec_contrast'][2],
                       transforms['spec_contrast'][3],
                       transforms['spec_contrast'][4],
                       transforms['spec_contrast'][5],
                       transforms['spec_contrast'][6],
                       transforms['tonnetz'][0],
                       transforms['tonnetz'][1],
                       transforms['tonnetz'][2],
                       transforms['tonnetz'][3],
                       transforms['tonnetz'][4],
                       transforms['tonnetz'][5]]

    return torch.FloatTensor(transforms_list).to(device='cuda')


# create one giant csv with all the tranforms data and the label of each wav file
def get_csv(csv_name, data_split, annotations_path):
    with open(csv_name, mode='w', newline='') as f:
        writer = csv.writer(f)

        for filename in data_split:
            try:
                print(str(csv_name) + ' THREAD: Applying transform to: ' + filename)
                transforms = apply_transforms(filename)
                writer.writerow([filename,
                                 transforms['mfccs'][0], transforms['mfccs'][1],
                                 transforms['mfccs'][2], transforms['mfccs'][3],
                                 transforms['mfccs'][4], transforms['mfccs'][5],
                                 transforms['mfccs'][6], transforms['mfccs'][7],
                                 transforms['mfccs'][8], transforms['mfccs'][9],
                                 transforms['mfccs'][10], transforms['mfccs'][11],
                                 transforms['mfccs'][12], transforms['mfccs'][13],
                                 transforms['mfccs'][14], transforms['mfccs'][15],
                                 transforms['mfccs'][16], transforms['mfccs'][17],
                                 transforms['mfccs'][18], transforms['mfccs'][19],
                                 transforms['mfccs'][20], transforms['mfccs'][21],
                                 transforms['mfccs'][22], transforms['mfccs'][23],
                                 transforms['mfccs'][24], transforms['mfccs'][25],
                                 transforms['mfccs'][26], transforms['mfccs'][27],
                                 transforms['mfccs'][28], transforms['mfccs'][29],
                                 transforms['mfccs'][30], transforms['mfccs'][31],
                                 transforms['mfccs'][32], transforms['mfccs'][33],
                                 transforms['mfccs'][34], transforms['mfccs'][35],
                                 transforms['mfccs'][36], transforms['mfccs'][37],
                                 transforms['mfccs'][38], transforms['mfccs'][39],
                                 transforms['chromagram'][0],
                                 transforms['chromagram'][1],
                                 transforms['chromagram'][2],
                                 transforms['chromagram'][3],
                                 transforms['chromagram'][4],
                                 transforms['chromagram'][5],
                                 transforms['chromagram'][6],
                                 transforms['chromagram'][7],
                                 transforms['chromagram'][8],
                                 transforms['chromagram'][9],
                                 transforms['chromagram'][10],
                                 transforms['chromagram'][11],
                                 transforms['melss'][0], transforms['melss'][1], transforms['melss'][2],
                                 transforms['melss'][3], transforms['melss'][4], transforms['melss'][5],
                                 transforms['melss'][6], transforms['melss'][7], transforms['melss'][8],
                                 transforms['melss'][9], transforms['melss'][10], transforms['melss'][11],
                                 transforms['melss'][12], transforms['melss'][13], transforms['melss'][14],
                                 transforms['melss'][15], transforms['melss'][16], transforms['melss'][17],
                                 transforms['melss'][18], transforms['melss'][19], transforms['melss'][20],
                                 transforms['melss'][21], transforms['melss'][22], transforms['melss'][23],
                                 transforms['melss'][24], transforms['melss'][25], transforms['melss'][26],
                                 transforms['melss'][27], transforms['melss'][28], transforms['melss'][29],
                                 transforms['melss'][30], transforms['melss'][31], transforms['melss'][32],
                                 transforms['melss'][33], transforms['melss'][34], transforms['melss'][35],
                                 transforms['melss'][36], transforms['melss'][37], transforms['melss'][38],
                                 transforms['melss'][39], transforms['melss'][40], transforms['melss'][41],
                                 transforms['melss'][42], transforms['melss'][43], transforms['melss'][44],
                                 transforms['melss'][45], transforms['melss'][46], transforms['melss'][47],
                                 transforms['melss'][48], transforms['melss'][49], transforms['melss'][50],
                                 transforms['melss'][51], transforms['melss'][52], transforms['melss'][53],
                                 transforms['melss'][54], transforms['melss'][55], transforms['melss'][56],
                                 transforms['melss'][57], transforms['melss'][58], transforms['melss'][59],
                                 transforms['melss'][60], transforms['melss'][61], transforms['melss'][62],
                                 transforms['melss'][63], transforms['melss'][64], transforms['melss'][65],
                                 transforms['melss'][66], transforms['melss'][67], transforms['melss'][68],
                                 transforms['melss'][69], transforms['melss'][70], transforms['melss'][71],
                                 transforms['melss'][72], transforms['melss'][73], transforms['melss'][74],
                                 transforms['melss'][75], transforms['melss'][75], transforms['melss'][76],
                                 transforms['melss'][77], transforms['melss'][78], transforms['melss'][79],
                                 transforms['melss'][80], transforms['melss'][81], transforms['melss'][82],
                                 transforms['melss'][83], transforms['melss'][84], transforms['melss'][85],
                                 transforms['melss'][86], transforms['melss'][87], transforms['melss'][88],
                                 transforms['melss'][89], transforms['melss'][90], transforms['melss'][91],
                                 transforms['melss'][92], transforms['melss'][93], transforms['melss'][94],
                                 transforms['melss'][95], transforms['melss'][96], transforms['melss'][97],
                                 transforms['melss'][98], transforms['melss'][99], transforms['melss'][100],
                                 transforms['melss'][101], transforms['melss'][102], transforms['melss'][103],
                                 transforms['melss'][104], transforms['melss'][105], transforms['melss'][106],
                                 transforms['melss'][107], transforms['melss'][108], transforms['melss'][109],
                                 transforms['melss'][110], transforms['melss'][111], transforms['melss'][112],
                                 transforms['melss'][113], transforms['melss'][114], transforms['melss'][115],
                                 transforms['melss'][116], transforms['melss'][117], transforms['melss'][118],
                                 transforms['melss'][119], transforms['melss'][120], transforms['melss'][121],
                                 transforms['melss'][122], transforms['melss'][123], transforms['melss'][124],
                                 transforms['melss'][125], transforms['melss'][126], transforms['melss'][127],
                                 transforms['spec_contrast'][0],
                                 transforms['spec_contrast'][1],
                                 transforms['spec_contrast'][2],
                                 transforms['spec_contrast'][3],
                                 transforms['spec_contrast'][4],
                                 transforms['spec_contrast'][5],
                                 transforms['spec_contrast'][6],
                                 transforms['tonnetz'][0],
                                 transforms['tonnetz'][1],
                                 transforms['tonnetz'][2],
                                 transforms['tonnetz'][3],
                                 transforms['tonnetz'][4],
                                 transforms['tonnetz'][5],
                                 get_label(filename)])
            except Exception:
                print(str(csv_name) + ' THREAD: ERROR AT ' + filename + '. CONTINUING ...')
                pass

    shutil.move(str(csv_name), str(root_dir + annotations_path))
    return


# prepare raw .wav files to the csv dataframe needed. This includes splitting the data into training and testing,
# applying the transforms, and saving as a csv. This function is used in the prepare_data.sh script
def prepare_dataset():

    annotations_path = '/voice_detect/data/annotations'
    voice_wavs = str(root_dir + '/voice_detect/data/voice/wav/')
    not_voice_wavs = str(root_dir + '/voice_detect/data/not_voice/wav/')

    print('Creating splits ...\n')
    train, test = create_splits(voice_wavs, not_voice_wavs)
    
    # start two threads, one to crate the training csv and one for the testing csv
    train_thread = threading.Thread(target=get_csv, args=('train.csv', train, annotations_path))
    test_thread = threading.Thread(target=get_csv, args=('test.csv', test, annotations_path))

    train_thread.start()
    test_thread.start()
    train_thread.join()
    test_thread.join()
