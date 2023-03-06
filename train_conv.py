import torch
import numpy as np
import pandas as pd
from ecg_bib import ECG_Dataset, ECG_CNN_Classifier, train
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from tqdm import tqdm

# Vou usar uma seed para que possamos reproduzir os resultados depois (E não sejamos desclassificados :p)
torch.manual_seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = ECG_Dataset('data/train_dccweek2023.h5',
                            'data/train_dccweek2023-labels.csv',
                             start = 0, end = 45000)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = ECG_Dataset('data/train_dccweek2023.h5',
                            'data/train_dccweek2023-labels.csv',
                             start = 45000, end = -1)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = ECG_CNN_Classifier().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

train(model, train_loader, val_loader, optimizer, criterion, 20, "models/ECG_CNN_Classifier")