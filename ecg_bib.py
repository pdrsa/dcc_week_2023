import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.metrics import f1_score
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ECG_Dataset(Dataset):
    def __init__(self, 
                tracings_file_path,
                labels_file_path,
                start = 0,
                end = -1):
        self.f = h5py.File(tracings_file_path, 'r')

        # Get tracings
        self.trace_ids = np.array(self.f['exam_id'])[start:end]
        self.tracings = self.f['tracings']

        # Defining start and end
        self.start = start
        self.end   = (end if end != -1 else len(self.tracings)-1)

        # Get labels
        labels_df = pd.read_csv(labels_file_path)
        self.labels    = {labels_df["exam_id"][i]:labels_df["classe"][i] for i in range(len(self.tracings))}

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, idx):
        # Get tracing
        tracing_idx = self.start + idx
        tracing = np.transpose(self.tracings[tracing_idx])
        
        # Get label
        label = self.labels[self.trace_ids[idx]]

        return tracing, label

class ECG_CNN_Classifier(nn.Module):
    def __init__(self, num_classes=7):
        super(ECG_CNN_Classifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=12, out_channels=32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(num_features=32)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(num_features=64)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(num_features=128)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(128*512, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Código geral de treinamento :)
# Em teoria, é só trocar o modelo que isso aqui deveria fazer a mágica acontecer
def train(model, train_loader, val_loader, optimizer, criterion, num_epochs, model_path, patience):
    print("Training!\nEpochs:",num_epochs,"\nPatience:", patience,"\n")
    
    best_val_loss = float('inf')
    epochs_since_last_improvement = 0

    for epoch in range(num_epochs):
        print("\nStarting epoch", epoch+1, "----------\n")
        # TREINAMENTO
        model.train()
        train_loss = 0.0
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)
        
        # VALIDAÇÃO
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # salva o modelo com menor perda na validação
        if val_loss < best_val_loss:
            torch.save(model.state_dict(), f"{model_path}/best_model.pth")
            best_val_loss = val_loss
            epochs_since_last_improvement = 0
        else:
            epochs_since_last_improvement += 1
        
        # verifica se deve parar o treinamento
        if epochs_since_last_improvement >= patience:
            print(f"No improvement for {patience} epochs. Stopping training.")
            break
        
        # salva o último modelo treinado
        torch.save(model.state_dict(), f"{model_path}/last_model.pth")

    print("Finished training!")

# Avaliação Genérica
def evaluate(model, dataloader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true += labels.cpu().numpy().tolist()
            y_pred += preds.cpu().numpy().tolist()
    return classification_report(y_true, y_pred, digits=4)