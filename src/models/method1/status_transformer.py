import sys
import os

project_root = os.path.abspath("/home/hajali/Desktop/Bargh_Ml_project/")
sys.path.insert(0, project_root)

from logs.logger import CustomLogger

logger = CustomLogger(name="transformer", log_gile='/home/hajali/Desktop/Bargh_Ml_project/logs/status_transfomer.log').get_logger()

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


class UnitStatusDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]


class StatusTransformer(nn.Module):

    def __init__(self, dataframe:pd.DataFrame, feature_cols: list[str], target_col: list[str], *args, **kwargs):
        
        super().__init__(*args, **kwargs)

        try:

            configs = self.load_configs(filename='/home/hajali/Desktop/Bargh_Ml_project/configs/status_transformer.yaml')
            logger.info(f'Configuration file successfully loaded.')

            self.input_dim = configs['input_dim']
            self.seq_len = configs['past_hours']
            self.output_dim = configs['future_hours']
            self.d_model = configs['emb_dim']
            self.num_classes = configs['output_dim']

            self.embedding = nn.Linear(self.input_dim, self.d_model)
            self.positional_encoding = nn.Parameter(torch.randn(self.seq_len, self.d_model))

            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=self.d_model, nhead=configs['num_heads'], dim_feedforward=configs['feedforward_dim'], dropout=configs['dropout'], batch_first=True),
                num_layers=configs['num_layers']
            )

            self.fc = nn.Linear(self.d_model * self.seq_len, self.output_dim * self.num_classes)
            self.softmax = nn.Softmax(dim=-1)

            logger.info(f'Transformer model succeessfully created. here is the structure of the model:\n{self}')

            self.features = feature_cols
            self.target = target_col

            le = LabelEncoder()
            label = le.fit_transform(dataframe['status'])
            dataframe.drop('status', axis=1, inplace=True)
            dataframe['status'] = label

            logger.info(f'Status feature changed into numeric type')

            dataframe = pd.get_dummies(dataframe, columns=['value'], drop_first=True)

            logger.info(f"Bar feature changed into Dummy type.")
            
            self.dataframe = dataframe

            logger.info(f"Model initiated completely")


        except Exception as e:
            logger.error(f'Could not initiate the class due to the below Exception:\n{e}')
    
    def prepare_data(self, name, code):
        df_copy = self.dataframe[(self.dataframe['name'] == name) & (self.dataframe['code'] == code)]

        logger.info(f"Data of name: {name} and code: {code} seperated from the original data to work on.")

        if len(df_copy.index) < 2:
            logger.warning(f'number of elements less than 2')
            return None
        
        scaler = StandardScaler()

        df_copy[self.features] = scaler.fit_transform(df_copy[self.features])

        logger.info(f"Input X and output Y scaled successfully")

        return df_copy
    
    def create_sequences(self, name, code, pastdays, futuredays):
        X, y = [], []

        data = self.prepare_data(name, code)

        for i in range(len(data) - pastdays - futuredays + 1):
            past_seq = data[self.features].iloc[i: i + pastdays].values
            future_seq = data[self.target[0]].iloc[i + pastdays: i + pastdays + futuredays].values
            if len(future_seq) != futuredays:
                print(f"Error at index {i}: Expected {futuredays}, got {len(future_seq)}")
            
            X.append(past_seq)
            y.append(future_seq)
        
        logger.info(f"Sequencde data prepared successfully.")
        
        return np.array(X), np.array(y)
    
    def forward(self, x):
        x = self.embedding(x)
        x += self.positional_encoding
        x = self.encoder(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        output = x.view(x.shape[0], self.output_dim, self.num_classes)
        return output
    
    def train_model(self, train_loader, val_loader, epochs=10, lr=0.001, weight_decay=1e-4, configs=None, save_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        train_loss = []
        test_loss = []

        for epoch in range(epochs):
            self.train()
            total_loss = 0
            for input, target in tqdm(train_loader, desc="Training Process", position=0, leave=True):
                input, target = input.to(device), target.to(device)
                optimizer.zero_grad()
                output = self.forward(input)
                loss = criterion(output.view(-1, self.num_classes), target.view(-1))
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
            
            scheduler.step()

            tqdm.write(f'Epoch {epoch+1}/{epochs}, Train Loss: {total_loss / len(train_loader):.4f} - Test Loss: {val_loss / len(val_loader):.4f}')
        
        
        if save_path is not None:
            save_path = self.save_nn(path=save_path, configs=configs, train_loss=train_loss, test_loss=test_loss)
        
        self.plot_training_process(epochs=epochs, train_loss=train_loss, test_loss=test_loss, save_path=save_path)
    
    def pipeline(self, name, code, lr=0.01, weight_decay=1e-4, epochs=10, save=False):

        configs = self.load_configs(filename='/home/hajali/Desktop/Bargh_Ml_project/configs/status_transformer.yaml')

        path = '/home/hajali/Desktop/Bargh_Ml_project/models/'

        X, y = self.create_sequences(name=name, code=code, pastdays=configs['past_hours'], futuredays=configs['future_hours'])

        dataset = UnitStatusDataset(X, y)

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=configs['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=configs['batch_size'], shuffle=False)
        if save:
            self.train_model(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs,
                lr=lr,
                weight_decay=weight_decay,
                save_path=f"{path}/{name}-{code}-Transformer#0/",
                configs=configs
            )
        else:
            self.train_model(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs,
                lr=lr,
                weight_decay=weight_decay
            )

    
    def predict(self, x, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.to(device)
        self.eval()
        predictions = []
        with torch.no_grad():
            for inputs, _ in x:
                inputs = inputs.to(device)
                outputs = torch.argmax(self.forward(inputs), dim=-1).cpu()
                predictions.append(outputs)
        return torch.cat(predictions, dim=0)
    
    def load_configs(self, filename):
        with open(filename, 'r') as f:
            return yaml.safe_load(f)
    
    def plot_training_process(self, epochs, train_loss, test_loss, save_path=None):
        plt.figure(figsize=(15, 8))
        plt.plot(range(1, epochs+1), train_loss, color='blue', label='Train loss', marker='o')
        plt.plot(range(1, epochs+1), test_loss, color='red', label='Test loss', marker='*')
        plt.xlabel("Epoch")
        plt.ylabel("Error")
        plt.title("Training Process plot")
        plt.legend()
        plt.grid(True)
        
        if save_path is not None:
            plt.savefig(f"{save_path}/Training_process.jpg")
            plt.close()
        else:
            plt.show()
    
    def save_nn(self, path, configs: dict, train_loss: list[int], test_loss: list[int]):
        try:
            counter = 1
            
            while os.path.exists(path=path):
                path = f"{path.split("#")[0]}#{counter}"
                counter += 1
            
            os.mkdir(path=path)

            torch.save(self.state_dict(), f"{path}/transformer_state_dict.pth")
            
            with open(f"{path}/Configuration.json", 'w') as f:
                json.dump(configs, f, indent=4)
            
            with open(f"{path}/Structure.html", 'w') as f:
                f.write(f"{self}")
                f.close()
            
            logger.info(msg=f"State of the Transformer saved successfully.")

            id = path.split('/')[-1].split('.')[0]

            bnchmrk.add_model(train_loss=train_loss, test_loss=test_loss, id=id, N=5, is_status_predictor=True)

            return path
        except Exception as e:
            logger.error(msg=f"Could not save the state of the transformer. Exception below occured:\n{e}")