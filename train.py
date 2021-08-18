import os
import warnings
import logging
import time
from datetime import date
import pandas as pd
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader
from ffnn_dataset import FFNN_dataset
from model import FFNN
from utils import get_accuracy

warnings.filterwarnings('ignore', category=UserWarning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 100
batch_size = 64
lr = 0.001
workers = 4

def main():
    # create new directory for the current training session
    today = date.today()
    today = str(today.strftime('%m-%d-%Y'))
    dir_ = os.path.join(os.getcwd(), 'saved_models/FFNN/train-' + today)

    if os.path.isdir(dir_):
        print(dir_, 'already exists. Continuing with training...' )
    else:
        print('Creating new dir: ', dir_)
        os.makedirs(dir_)

    log_file_name = 'FFNN-' + today +'.log'
    logging.basicConfig(filename=os.path.join(dir_, log_file_name),
                        filemode='w',
                        format='%(asctime)s: %(message)s',
                        level=logging.INFO)
    # get training set
    df = pd.read_csv('data/annotations/train.csv', header=None)
    
    # encode labels
    encode = {'voice' : 1, 'not_voice' : 0}
    df.iloc[:, 195].replace(encode, inplace=True)

    # remove file names
    df.drop(df.columns[0], axis=1, inplace=True)

    # seperate data and labels
    x = df.iloc[:, 0: -1]
    y = df.iloc[:, -1]
    
    # get dataloader for training
    train_data = FFNN_dataset(torch.FloatTensor(x.values), torch.FloatTensor(y.values))
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    
    model = FFNN()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # for each epoch
    for epoch in range(epochs):

        model.train()
        start = time.time()
        epoch_loss = 0
        epoch_accuracy = 0
        epoch_steps = 0

        for i, (x, y) in enumerate(train_loader):

            x, y = x.to(device), y.to(device, dtype=torch.int64)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y) #y.float()
            epoch_accuracy = get_accuracy(prediction, y)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_steps += 1
            start = time.time()

        # print status onto terminal and log file
        print('Epoch: [%d/%d] | Loss: %.3f | Accuracy: %.3f' % (epoch+1,
                                                                epochs,
                                                                epoch_loss,
                                                                epoch_accuracy))

        logging.info('Epoch: [%d/%d] | Loss: %.3f | Accuracy: %.3f' % (epoch+1,
                                                                       epochs,
                                                                       epoch_loss,
                                                                       epoch_accuracy))
        # save model
        model_file_name = 'FFNN-' + today + '.pt'
        torch.save(model.state_dict(), os.path.join(dir_, model_file_name))

if __name__ == '__main__':
    main()
