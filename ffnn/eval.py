import os
import sys
import argparse
import warnings
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from model import FFNN
from dataset import FFNN_dataset

warnings.filterwarnings('ignore', category=UserWarning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()
parser.add_argument('model_path', type=str, help='Path to feed forward network you wish to use')
args = parser.parse_args()

def main():

    # set parameters
    batch_size = 32
    nb_classes = 2
    pred_list = torch.zeros(0, dtype=torch.long, device='cpu')
    ground_truth = torch.zeros(0, dtype=torch.long, device='cpu')
    root_dir = str(os.path.abspath(os.path.join(os.getcwd(),os.pardir)))

    # get model
    model_path = str(root_dir + '/saved_models/' + args.model_path)
    model = FFNN()
    model.load_state_dict(torch.load(model_path), strict=False)
    model.to(device)
    model.eval()

    # get testing dataset
    data_path = str(root_dir + '/data/annotations/test.csv')
    df = pd.read_csv(data_path, header=None)
    
    # encode labels
    encode = {'voice' : 1, 'not_voice' : 0}
    df.iloc[:, 195].replace(encode, inplace=True)

    # remove file names
    df.drop(df.columns[0], axis=1, inplace=True)
    
    # seperate data and labels
    x = df.iloc[:, 0:-1]
    y = df.iloc[:, -1:]

    # get testing dataset
    test_data = FFNN_dataset(torch.FloatTensor(x.values), torch.FloatTensor(y.values))
    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                              batch_size=batch_size,
                                              shuffle=True)
    # start testing
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, preds = torch.max(outputs.data, 1)
            
            pred_list = torch.cat([pred_list, preds.view(-1).cpu()])
            ground_truth = torch.cat([ground_truth, y.view(-1).cpu()])

    # accuracy report
    print('\nAccuracy Score:')
    print(accuracy_score(ground_truth.numpy(), pred_list.numpy()))

    # confusion matrix
    print('\nConfusion Matrix:')
    conf_mat = confusion_matrix(ground_truth.numpy(), pred_list.numpy())
    print(conf_mat)

    # per-class accuracy
    print('\nPer-Class Accuracy:')
    print(100 * conf_mat.diagonal() / conf_mat.sum(1))
    
    # classification report
    print('\nClassification Report:')
    print(classification_report(ground_truth.numpy(), pred_list.numpy()))

if __name__ == '__main__':
    main()
