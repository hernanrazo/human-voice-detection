import os
import argparse
import warnings
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import torch
import torchvision
from torchvision.transforms import transforms
from model import CNN

warnings.filterwarnings('ignore', category=UserWarning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root_dir = str(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
parser = argparse.ArgumentParser()
parser.add_argument('model_path', type=str, help='Path to convolutional neural network you wish to use')
args = parser.parse_args()

def main():

    # set parameters
    batch_size = 16
    workers = 4
    nb_classes = 2
    pred_list = torch.zeros(0, dtype=torch.long, device='cpu')
    ground_truth = torch.zeros(0, dtype=torch.long, device='cpu')

    # get model
    model_path = str(root_dir + '/saved_models/' + args.model_path)
    model = CNN()
    model.load_state_dict(torch.load(model_path), strict=False)
    model.to(device)
    model.eval()

    # set basic transforms. Spectrograms have to look a certain way so rotations, flips, and other
    # transforms do not make sense in this application
    transform = {'test' : transforms.Compose([transforms.Resize([32, 32]), transforms.ToTensor()])}

    # get testing dataset
    test_data = torchvision.datasets.ImageFolder(root_dir + '/data/plots/test/',
                                                 transform=transform['test'])

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

    # accuracy score
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
