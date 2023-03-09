import torch
import numpy as np
import pandas as pd
from ecg_bib import ECG_CNN_Classifier, train, evaluate
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from tqdm import tqdm
import h5py

MODEL_NAME = "ECG_CNN_Binary_Classifier"

class ECG_Dataset(Dataset):
    def __init__(self, 
                tracings_file_path,
                labels_file_path,
                start = 0,
                end = -1):
        self.f = h5py.File(tracings_file_path, 'r')

        # Get tracings
        self.trace_ids = np.array(self.f['exam_id'])
        self.tracings = self.f['tracings']

        # Defining start and end
        self.start = start
        self.end   = (end if end != -1 else len(self.tracings)-1)

        # Get labels
        labels_df = pd.read_csv(labels_file_path)
        self.labels    = {labels_df["exam_id"][i]:labels_df["classe"][i] for i in range(len(self.tracings))}

        self.indexes = []

        count = [0, 0]
        # Balancing dataset
        for i in range(self.start, self.end):
            c = self.get_label(i)
            if(c == 0 and count[0] - count[1] > count[0]/2): continue
            self.indexes.append(i)
            count[c] += 1


    def get_label(self, idx):
        return min(self.labels[self.trace_ids[idx]], 1)

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        idx = self.indexes[idx]

        # Get tracing
        tracing = np.transpose(self.tracings[idx])
        
        # Get label
        label = self.get_label(idx)

        return tracing, label

# Vou usar uma seed para que possamos reproduzir os resultados depois (E não sejamos desclassificados :p)
torch.manual_seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = ECG_Dataset('data/train_dccweek2023.h5',
                            'data/train_dccweek2023-labels.csv',
                             start = 0, end = 46000)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = ECG_Dataset('data/train_dccweek2023.h5',
                            'data/train_dccweek2023-labels.csv',
                             start = 46000, end = -1)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = ECG_CNN_Classifier(num_classes=2).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
train(model, train_loader, val_loader, optimizer, criterion, 20, "models/" + MODEL_NAME, 3)

# Testing
test_report = evaluate(model, val_loader, device)
with open('results/' + MODEL_NAME + '.txt', 'w') as f:
    f.write(test_report)
print(test_report)
