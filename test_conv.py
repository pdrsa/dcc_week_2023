import torch
from torch.utils.data import DataLoader
from ecg_bib import evaluate, ECG_Dataset, ECG_CNN_Classifier, evaluate_binary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = ECG_CNN_Classifier().to(device)
model_path = "models/ECG_CNN_Classifier_1.1/best_model.pth"
model.load_state_dict(torch.load(model_path))
test_dataset = ECG_Dataset('data/train_dccweek2023.h5',
                            'data/train_dccweek2023-labels.csv',
                             start = 45000, end = -1)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

test_report    = evaluate(model, test_loader, device)
test_report_bi = evaluate_binary(model, test_loader, device)

with open('test_report.txt', 'w') as f:
    f.write(test_report)
print(test_report)

with open('test_report_bi.txt', 'w') as f:
    f.write(test_report_bi)
print(test_report_bi)