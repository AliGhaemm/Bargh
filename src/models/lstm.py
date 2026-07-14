import sys
import os

project_root = os.path.abspath("/home/hajali/Desktop/Bargh_Ml_project/")
sys.path.insert(0, project_root)

from logs.logger import CustomLogger

logger = CustomLogger(name="LSTM", log_gile='/home/hajali/Desktop/Bargh_Ml_project/logs/status_lstm.log').get_logger()

from benchmark import Benchmark

bnchmrk = Benchmark()

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm
import json
import time
from status_transformer import UnitStatusDataset

class StatusLSTM(nn.Module):
    def __init__(self, dataframe: pd.DataFrame, feature_cols: list[str], target_col: list[str], *args, **kwargs):
        super().__init__(*args, **kwargs)

        configs = self.load_configs(filename='/home/hajali/Desktop/Bargh_Ml_project/configs/status_transformer.yaml')

        self.input_dim = configs['input_dim']
        self.output_dim = configs['future_hours']
        self.hidden_size = 64
        self.num_classes = configs['output_dim']
        self.num_layers = configs['num_layers']

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=configs['dropout'],
            batch_first=True
        )

        self.fc = nn.Linear(self.hidden_size, self.output_dim * self.num_classes)

        self.feature_cols = feature_cols
        self.target_col = target_col

        logger.info(msg=f"LSTM model initiated successfully. the structure of the model is:\n{self}")

        le = LabelEncoder()
        label = le.fit_transform(dataframe['status'])
        dataframe.drop('status', axis=1, inplace=True)
        dataframe['status'] = label

        logger.info(msg=f"Label encoder fitted on Status.")

        dataframe = pd.get_dummies(dataframe, columns=['value'], drop_first=True)
        
        self.dataframe = dataframe

        torch.set_float32_matmul_precision("high")
    
    def prepare_data(self, name, code):
        try:
            df_copy = self.dataframe[(self.dataframe['name'] == name) & (self.dataframe['code'] == code)]

            if len(df_copy.index) < 2:
                print(f'numbeer of elements less than 2')
                return None
            
            scaler = StandardScaler()

            df_copy[self.feature_cols] = scaler.fit_transform(df_copy[self.feature_cols])

            logger.info(msg=f"Data with name: {name} and code: {code} splitted. also standard scaler applied on data")

            return df_copy
        except Exception as e:
            logger.error(msg=f"Could not prepare data. Exception below occured:\n{e}")
            return None
    
    def create_sequences(self, name, code, pastdays, futuredays):
        try:
            X, y = [], []

            data = self.prepare_data(name, code)

            for i in range(len(data) - pastdays - futuredays + 1):
                past_seq = data[self.feature_cols].iloc[i: i + pastdays].values
                future_seq = data[self.target_col[0]].iloc[i + pastdays: i + pastdays + futuredays].values
                if len(future_seq) != futuredays:
                    print(f"Error at index {i}: Excpected {futuredays} git {len(future_seq)}")
                
                X.append(past_seq)
                y.append(future_seq)
            
            logger.info(msg=f"Sequence of data created. Length of history is: {pastdays} and length of prediction is: {futuredays}")
            return np.array(X), np.array(y)
        except Exception as e:
            logger.error(msg=f"Could not create sequence of data. Exception below occured:\n{e}")
            return None
    
    def forward(self, x):
        self.lstm.flatten_parameters()
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
    
    def train_model(self, train_loader, val_loader, epochs=10, lr=0.01, weight_decay=0.01, configs=None, save_path=None, device="cuda" if torch.cuda.is_available() else "cpu"):
        try:
            torch.set_num_threads(8)
            self.to(device=device)
            self = torch.compile(self)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
            optimizer_settings = {
                "type": "Adam",
                "learning rate": lr,
                "betas": optimizer.param_groups[0]['betas'],
                "eps": optimizer.param_groups[0]["eps"],
                "weight decay": weight_decay
            }
            # optimizer = optim.RMSprop(self.parameters(), lr=lr, alpha=0.99, eps=1e-8, weight_decay=weight_decay)
            # optimizer_settings = {
            #     "type": "RMSprop",
            #     "learning rate": lr,
            #     "alpha": optimizer.param_groups[0]["alpha"],
            #     "eps": optimizer.param_groups[0]["eps"],
            #     "weight decay": weight_decay
            # }
            
            train_loss = []
            test_loss = []
            start_time = time.time()
            for epoch in range(epochs):
                self.train()
                total_loss = 0
                for input, target in tqdm(train_loader, desc="Training Process", position=0, leave=True):
                    input, target = input.to(device), target.to(device)
                    optimizer.zero_grad()
                    outputs = self.forward(input)
                    loss = criterion(outputs.view(-1, self.num_classes), target.view(-1))
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                train_loss.append(total_loss / len(train_loader))
            
                self.eval()
                val_loss = 0
                with torch.no_grad():
                    for input, target in val_loader:
                        input, target = input.to(device), target.to(device)
                        output = self.forward(input)
                        loss = criterion(output.view(-1, self.num_classes), target.view(-1))
                        val_loss += loss.item()
                    test_loss.append(val_loss / len(val_loader))
                
                tqdm.write(f'Epoch {epoch+1}/{epochs}, Train Loss: {total_loss / len(train_loader):.4f} - Test Loss: {val_loss / len(val_loader):.4f}')
            
            configs['epochs'] = epochs
            configs['time consuming'] = time.time() - start_time
            if save_path is not None:
                save_path = self.save_lstm(path=save_path, configs=configs, train_loss=train_loss, test_loss=test_loss, optimizer_settings=optimizer_settings)
            
            self.plot_training_process(epochs=epochs, train_loss=train_loss, test_loss=test_loss, save_path=save_path)
        except Exception as e:
            logger.error(msg=f"Training stoped. Exception below occured:\n{e}")
    
    def pipeline(self, name, code, lr=0.01, epochs=10, weight_decay=1e-2, save=False):
        
        configs = self.load_configs(filename='/home/hajali/Desktop/Bargh_Ml_project/configs/status_transformer.yaml')

        path = "/home/hajali/Desktop/Bargh_Ml_project/models/"

        X, y = self.create_sequences(name=name, code=code, pastdays=configs['past_hours'], futuredays=configs['future_hours'])

        dataset = UnitStatusDataset(X, y)

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)
        if save:
            self.train_model(
                train_loader,
                val_loader,
                epochs,
                lr,
                weight_decay=weight_decay,
                save_path=f"{path}/{name}-{code}-LSTM#0",
                configs=configs
            )
        else:
            self.train_model(
                train_loader,
                val_loader,
                epochs,
                lr,
                weight_decay=weight_decay
            )
    
    def predict(self, x, device="cuda" if torch.cuda.is_available() else "cpu"):
        try:
            self.to(device)
            self.eval()
            predictions = []

            with torch.no_grad():
                for inputs, _ in x:
                    inputs = inputs.to(device)
                    outputs = torch.argmax(self.forward(inputs, dim=-1)).cpu()
                    predictions.append(outputs)
            
            return torch.cat(predictions, dim=0)
        except Exception as e:
            logger.error(msg=f"Could not predict the input data. Exception below occured:\n{e}")

    def load_configs(self, filename):
        with open(filename, 'r') as f:
            return yaml.safe_load(f)
    
    def plot_training_process(self, epochs, train_loss, test_loss, save_path=None):
        plt.figure(figsize=(15, 8))
        plt.plot(range(1, epochs+1), train_loss, color='blue', label='Train Loss', marker='o')
        plt.plot(range(1, epochs+1), test_loss, label='Test Loss', color='red', marker='*')
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.title('Training Process Plot')
        plt.legend()
        plt.grid(True)

        if save_path is not None:
            plt.savefig(f"{save_path}/Training_process.jpg")
            plt.close()
        else:
            plt.show()
    
    def save_lstm(self, path, configs: dict, train_loss, test_loss, optimizer_settings:dict):
        try:
            counter = 1
            
            while os.path.exists(path=path):
                path = f"{path.split("#")[0]}#{counter}"
                counter += 1
            
            os.mkdir(path=path)

            torch.save(self.state_dict(), f"{path}/LSTM_state_dict.pth")
            
            with open(f"{path}/Configuration.json", 'w') as f:
                json.dump(configs, f, indent=4)
            
            with open(f"{path}/Structure.txt", 'w') as f:
                f.write(f"{self}")
                f.close()
            
            with open(f"{path}/Optimizer.json", 'w') as f:
                json.dump(optimizer_settings, f, indent=4)
            
            logger.info(msg=f"State of the Transformer saved successfully.")

            id = path.split('/')[-1].split('.')[0]

            bnchmrk.add_model(train_loss=train_loss, test_loss=test_loss, id=id, N=5, is_status_predictor=True)

            return path
        except Exception as e:
            logger.error(msg=f"Could not save the state of the transformer. Exception below occured:\n{e}")
            return path