Voice Detection
===

This project is a binary classification problem of audio data that aims to classify human voices from audio recordings. This project uses a feed forward neural network and a convolutional neural network where both networks work together in a voting classifier fashion to increase accuracy on never before seen data.  

All neural networks were implemented in PyTorch, audio utilities were implemented using Librosa, and the whole project is written in Python 3.8.5. The data for this project was obtained from the [Urban Sound Classification practice problem from Analytics Vidhya](https://datahack.analyticsvidhya.com/contest/practice-problem-urban-sound-classification/) and the voice recordings data was taken from the [LibreSpeech ASR Corpus.](https://www.openslr.org/12)  

Data Preparation
---

To get the dataset, download both sections from the above links. I specifically rearranged the dataset by placing all non-voice recordings in one directory and voice recordings in another:  

```bash
├── data
│   ├── voice
│   │   ├── rec1.wav
│   │   ├── rec2.wav
│   │   ├── ...
│   ├── not_voice
│   │   ├── rec1.wav
│   │   ├── rec2.wav
│   │   ├── ...
```

The `librespeech.py` script can be used to transform the LibreSpeech dataset to the above file structure. All other files that remain can be deleted. Ignore this if you are not using the LibreSpeech dataset.  

The feed forward neural network uses the Mel-frequency cepstral coefficient, chromagram, mel-scaled spectrogram, spectral contrast, and the tonal centroid features as input. The Librosa python library is used to obtain all these calculations in the `apply_transforms()` function in the ffnn utils. The `prep_ffnn_data.sh` shell script can be run to obtain all these features for each .wav file in one giant csv file. One entry will look like:
```
/home/hernanrazo/pythonProjects/voice_detect/data/voice/wav/6476-57446-         0035.wav,-345.36868,111.94338,-14.379323,50.732822,-13.364564,-0.81879413,-     8.603202,-11.860127,0.3794937,-11.088642,-0.5936078,-8.392053,-4.562349,2.      390204,2.1774976,-1.8160796,-0.79248935,1.3244591,-3.649716,-2.789777,-3.       483583,-2.1718845,-7.1207843,-4.646477,-2.145171,-5.4034863,1.1288224,2.        6650674,6.1765018,8.234708,5.759141,8.06815,8.3969555,6.328495,5.646016,4.      1650767,2.2295291,1.0025103,-0.5408073,-1.0010004,0.41421363,0.3886434,0.       39528078,0.4243577,0.44586822,0.46885604,0.5323573,0.6273547,0.6136627,0.       5895413,0.5854901,0.5013139,0.030495275,0.025378285,0.03576337,0.02598723,      0.01064823,0.111760125,2.9451604,9.584426,10.904745,6.3265867,0.767396,0.       14292528,0.19490032,0.85179096,5.0548906,16.62874,16.24418,15.094028,11.        092577,5.3894925,1.5112005,0.42363763,0.42279428,0.6986573,1.1323513,1.         4325676,1.5941,1.5745226,0.6854819,0.22188246,0.18639795,0.25544456,0.          37152404,0.18624847,0.18722062,0.24387933,0.15841863,0.2312459,0.12505762,      0.0896525,0.06176768,0.033809755,0.06561177,0.11577808,0.08457274,0.            056273155,0.046364002,0.03207818,0.026625242,0.033034343,0.047393396,0.         039878745,0.030250499,0.035353974,0.04822752,0.088709675,0.08721649,0.          042465515,0.050014295,0.043818373,0.025141228,0.026777223,0.05408083,0.         054930124,0.042547297,0.027444469,0.015712438,0.013818915,0.014640613,0.        017465897,0.014250277,0.019179987,0.021202719,0.040190093,0.024158962,0.        020575762,0.020575762,0.019340117,0.01956742,0.0073476452,0.012725379,0.        016156813,0.007385745,0.008848519,0.0073545426,0.0060878447,0.007746159,0.      011803486,0.00961405,0.011231303,0.012259503,0.008804519,0.008680856,0.         008589337,0.0158784,0.015149302,0.0085100345,0.007378557,0.009641291,0.         0066143535,0.0060657472,0.003713564,0.0021371976,0.0019380879,0.0013283227,     0.0012585397,0.0009210656,0.0008644426,0.0008410996,0.00046661997,0.            00033427356,0.00020592447,5.9694554e-05,1.1552337e-05,2.8310662e-06,4.          4607148e-07,4.411787e-08,1.8092163e-09,1.2725149e-10,1.8920865e-10,1.           2470465e-10,9.163159e-11,1.8638106e-10,2.1313133e-10,7.265922e-10,3.            1799022e-10,8.475092e-10,7.542699e-10,1.6082426e-10,14.965268185360362,18.      193254004666265,20.9569399219138,17.267001240479917,18.13293584976544,19.       771650662276468,41.46849881683453,-0.019441445021252834,0.                      0061759247744320065,0.05519930844766153,0.004244935825248924,-0.                004941592482226379,-0.005592662805732028,voice
``` 
Each individual value gets its own cell and the label (voice/not_voice) gets attached to the end.  

The convolutional neural network recieves an image of the recording's Mels-spectrogram as input. Each image is obtained using the Librosa library. The `prep_cnn_data.sh` shell script can be used to obtain a spectrogram for each audio recording. Example:  

<img src="https://github.com/hernanrazo/human-voice-detection/blob/master/example.jpg" width="400" height="400"> 

All images are later scaled to 32x32 and transformed to tensors for training.  

Both shell scripts automatically split the data into training and testing sets in a 80/20 ratio.  

Neural Network Architectures
---
The feed forward neural network is comprised of 3 linear layers with ReLU activation and dropout. The last linear layer returns 2 values, one for each class, and passes the result to a sigmoid activation function for the final output. For training, cross entropy loss is used along with an Adam optimizer. Training goes on for 70 epochs with a batch size of 32 and learning rate of 0.005.  

The convolutional neural network is comprised of 3 convolution layers with max pooling, batch normalization, and dropout. There are also 2 linear layers at the end with ReLU activation and dropout. The final linear layer does not have dropout and the output layer gives two values, one for each class. No activation layers are implemented since the cross entropy loss function has softmax built in. For training, I used cross entropy loss and Adam optimization. Training goes on for 250 epochs with batch size of 16 and a learning rate of 0.01.  

Performance
---
Performance for both models can be measured using the `eval.py` script in each model's respective directory. The script takes the file path to the model as a command line argument. These scripts return the accuracy score, confusion matrix, per-class accuracy, and classification report for the model in question.  

For the feed forward neural network:
```
python eval.py CNN/train-08-29-2021/CNN-08-29-2021.pt
``` 
And for the convolutional neural network:
```
python eval.py CNN/train-04-20-2021/CNN-04-20-2021.pt
```

For the feed forward neural network, I obtained 0.9758 accuracy. For the convolutional neural network, I obtianed 0.977 accuracy.

Voting Classifier Implementation
---
The voting classifier is implemented in the `main.py` script. Similar to the evaluation scripts, this script takes two command line arguments, the path to the feed forward neural network and the path to the convolutional neural network. Example:
```
python main.py FFNN/train-07-14-2021/FFNN-07-14-2021.pt CNN/train-04-20-2021/CNN-04-20-2021.pt
```
The script first calculates the needed transformations for the feed forward neural network and then creates the spectrogram for the convolutional neural network. The spectrogram is then resized to a 32x32 image and then converted to a tensor. Each input is passed to its respective neural network and an inference score is returned. If both networks return an inference above 0.85 for the recording being a voice, the result is deemed a voice. Any lower scoring is deemed not a voice.

Sources and helpful Links
---
https://www.telusinternational.com/articles/what-is-audio-classification#:~:text=Audio%20classification%20is%20the%20process,and%20text%20to%20speech%20applications.  
https://stackoverflow.com/questions/53290306/confusion-matrix-and-test-accuracy-for-pytorch-transfer-learning-tutorial  
https://librosa.org/doc/latest/index.html  
https://datahack.analyticsvidhya.com/contest/practice-problem-urban-sound-classification/  
https://www.openslr.org/12
