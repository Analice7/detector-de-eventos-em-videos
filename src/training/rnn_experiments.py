import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import os
import json
from pathlib import Path
from datetime import datetime

# Importando suas classes existentes
from train import PoseDataset, Config

class ExperimentConfig:
    SEED = 42
    BATCH_SIZE = 32
    NUM_LAYERS = 2
    DROPOUT = 0.3
    LEARNING_RATE = 0.001
    EPOCHS = 50
    SEQUENCE_LENGTH = 64    # (t) número de timesteps - modificado para 64
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Parâmetros para os experimentos
    RNN_TYPES = ['lstm', 'gru', 'rnn']  # (r) tipos de RNN a serem testados
    HIDDEN_SIZES = [32, 64, 128, 256, 512]  # (e) tamanhos do estado escondido
    
    # Diretório para salvar resultados
    RESULTS_DIR = 'experiment_results'

# Modelo RNN modificado para aceitar hidden_size como parâmetro
class PoseRNN(nn.Module):
    def __init__(self, input_size, rnn_type='lstm', hidden_size=512, num_layers=2, dropout=0.3):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.hidden_size = hidden_size
        
        # Camada RNN
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout
            )
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout
            )
        else:  # RNN simples
            self.rnn = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout
            )
        
        # Camada de classificação
        self.fc = nn.Linear(hidden_size, 2)  # 2 classes (normal, assault)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        out, _ = self.rnn(x)
        out = out[:, -1, :]  # Pega apenas o último timestep
        out = self.dropout(out)
        out = self.fc(out)
        return out

# Funções para treinamento e avaliação
def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, device, 
                experiment_name, checkpoint_dir):
    """Treina o modelo e salva os checkpoints e métricas."""
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_acc = 0
    metrics = {
        'train_losses': [],
        'val_losses': [],
        'val_accs': [],
        'epochs': []
    }
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        
        # Loop de treino
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.squeeze())
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Validação
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        train_loss = running_loss / len(train_loader)
        
        # Salvar métricas
        metrics['train_losses'].append(train_loss)
        metrics['val_losses'].append(val_loss)
        metrics['val_accs'].append(val_acc)
        metrics['epochs'].append(epoch + 1)
        
        # Salvar melhor modelo
        if val_acc > best_acc:
            best_acc = val_acc
            model_path = os.path.join(checkpoint_dir, f"{experiment_name}_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, model_path)
        
        print(f'Epoch {epoch+1}/{epochs} | '
              f'Train Loss: {train_loss:.4f} | '
              f'Val Loss: {val_loss:.4f} | '
              f'Val Acc: {val_acc:.2%}')
    
    # Salvar métricas em JSON
    metrics_path = os.path.join(checkpoint_dir, f"{experiment_name}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)
    
    return metrics

def evaluate(model, loader, criterion, device):
    """Avalia o modelo em um conjunto de dados."""
    model.eval()
    total = 0
    correct = 0
    running_loss = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels.squeeze())
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.squeeze()).sum().item()
    
    return running_loss / len(loader), correct / total

def test_model(model, test_loader, criterion, device, checkpoint_path=None):
    """Testa o modelo e retorna métricas detalhadas."""
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    all_preds = []
    all_labels = []
    running_loss = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels.squeeze())
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.squeeze().cpu().numpy())
    
    # Calcular métricas
    test_loss = running_loss / len(test_loader)
    test_report = classification_report(all_labels, all_preds, target_names=['normal', 'assault'], output_dict=True, zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    return test_loss, test_report, conf_matrix

def plot_training_curves(metrics, title, save_path=None):
    """Plota curvas de treinamento com base nas métricas salvas."""
    plt.figure(figsize=(12, 5))
    
    # Plot de loss
    plt.subplot(1, 2, 1)
    plt.plot(metrics['epochs'], metrics['train_losses'], label='Train Loss')
    plt.plot(metrics['epochs'], metrics['val_losses'], label='Val Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot de acurácia
    plt.subplot(1, 2, 2)
    plt.plot(metrics['epochs'], metrics['val_accs'], label='Val Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_confusion_matrix(conf_matrix, title, save_path=None):
    """Plota matriz de confusão."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['normal', 'assault'],
                yticklabels=['normal', 'assault'])
    plt.title(f'{title} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def run_type_experiment(input_size, train_loader, val_loader, test_loader, config):
    """Executa experimentos comparando diferentes tipos de RNN com hidden_size fixo."""
    
    results_dir = os.path.join(config.RESULTS_DIR, 'rnn_types')
    os.makedirs(results_dir, exist_ok=True)
    
    # Dicionário para armazenar os resultados
    test_results = {}
    
    for rnn_type in config.RNN_TYPES:
        print(f"\n{'='*20} Treinando {rnn_type.upper()} {'='*20}")
        
        # Criar modelo
        model = PoseRNN(
            input_size=input_size,
            rnn_type=rnn_type,
            hidden_size=512,  # Modificado para 512
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT
        ).to(config.DEVICE)
        
        # Loss e otimizador
        criterion = nn.CrossEntropyLoss() 
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        
        # Nome do experimento
        experiment_name = f"{rnn_type}_h512"  # Modificado para h512
        checkpoint_dir = os.path.join(results_dir, experiment_name)
        
        # Treinar
        metrics = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            epochs=config.EPOCHS,
            device=config.DEVICE,
            experiment_name=experiment_name,
            checkpoint_dir=checkpoint_dir
        )
        
        # Plotar curvas de treinamento
        plot_path = os.path.join(checkpoint_dir, f"{experiment_name}_training_curves.png")
        plot_training_curves(metrics, f"{rnn_type.upper()} Training", save_path=plot_path)
        
        # Testar o melhor modelo
        checkpoint_path = os.path.join(checkpoint_dir, f"{experiment_name}_best.pth")
        test_loss, test_report, conf_matrix = test_model(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=config.DEVICE,
            checkpoint_path=checkpoint_path
        )
        
        # Salvar matriz de confusão
        conf_matrix_path = os.path.join(checkpoint_dir, f"{experiment_name}_confusion_matrix.png")
        plot_confusion_matrix(conf_matrix, f"{rnn_type.upper()} Test", save_path=conf_matrix_path)
        
        # Salvar resultados
        test_results[rnn_type] = {
            'test_loss': test_loss,
            'test_accuracy': test_report['accuracy'],
            'test_f1_macro': test_report['macro avg']['f1-score'],
            'normal_f1': test_report['normal']['f1-score'],
            'assault_f1': test_report['assault']['f1-score']
        }
        
        # Salvar o relatório detalhado
        report_path = os.path.join(checkpoint_dir, f"{experiment_name}_test_report.json")
        with open(report_path, 'w') as f:
            json.dump(test_report, f)
    
    # Criar tabela comparativa
    results_df = pd.DataFrame(test_results).T
    results_df = results_df.round(4)
    results_df.index.name = 'RNN Type'
    
    # Salvar a tabela
    results_df.to_csv(os.path.join(results_dir, 'rnn_type_comparison.csv'))
    
    # Plotar comparação
    plt.figure(figsize=(12, 6))
    results_df.plot(kind='bar', figsize=(12, 6))
    plt.title('Comparação entre Tipos de RNN')
    plt.ylabel('Valor')
    plt.xlabel('Tipo de RNN')
    plt.xticks(rotation=0)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'rnn_type_comparison.png'))
    plt.show()
    
    return results_df

def run_hidden_size_experiment(input_size, train_loader, val_loader, test_loader, config, best_rnn_type):
    """Executa experimentos comparando diferentes tamanhos de hidden state para o melhor tipo de RNN."""
    
    results_dir = os.path.join(config.RESULTS_DIR, 'hidden_sizes')
    os.makedirs(results_dir, exist_ok=True)
    
    # Dicionário para armazenar os resultados
    test_results = {}
    
    for hidden_size in config.HIDDEN_SIZES:
        print(f"\n{'='*20} Treinando {best_rnn_type.upper()} com hidden_size={hidden_size} {'='*20}")
        
        # Criar modelo
        model = PoseRNN(
            input_size=input_size,
            rnn_type=best_rnn_type,
            hidden_size=hidden_size,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT
        ).to(config.DEVICE)
        
        # Loss e otimizador
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        
        # Nome do experimento
        experiment_name = f"{best_rnn_type}_h{hidden_size}"
        checkpoint_dir = os.path.join(results_dir, experiment_name)
        
        # Treinar
        metrics = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            epochs=config.EPOCHS,
            device=config.DEVICE,
            experiment_name=experiment_name,
            checkpoint_dir=checkpoint_dir
        )
        
        # Plotar curvas de treinamento
        plot_path = os.path.join(checkpoint_dir, f"{experiment_name}_training_curves.png")
        plot_training_curves(metrics, f"{best_rnn_type.upper()} h={hidden_size} Training", save_path=plot_path)
        
        # Testar o melhor modelo
        checkpoint_path = os.path.join(checkpoint_dir, f"{experiment_name}_best.pth")
        test_loss, test_report, conf_matrix = test_model(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=config.DEVICE,
            checkpoint_path=checkpoint_path
        )
        
        # Salvar matriz de confusão
        conf_matrix_path = os.path.join(checkpoint_dir, f"{experiment_name}_confusion_matrix.png")
        plot_confusion_matrix(conf_matrix, f"{best_rnn_type.upper()} h={hidden_size} Test", save_path=conf_matrix_path)
        
        # Salvar resultados
        test_results[hidden_size] = {
            'test_loss': test_loss,
            'test_accuracy': test_report['accuracy'],
            'test_f1_macro': test_report['macro avg']['f1-score'],
            'normal_f1': test_report['normal']['f1-score'],
            'assault_f1': test_report['assault']['f1-score']
        }
        
        # Salvar o relatório detalhado
        report_path = os.path.join(checkpoint_dir, f"{experiment_name}_test_report.json")
        with open(report_path, 'w') as f:
            json.dump(test_report, f)
    
    # Criar tabela comparativa
    results_df = pd.DataFrame(test_results).T
    results_df = results_df.round(4)
    results_df.index.name = 'Hidden Size'
    
    # Salvar a tabela
    results_df.to_csv(os.path.join(results_dir, f'{best_rnn_type}_hidden_size_comparison.csv'))
    
    # Plotar comparação
    plt.figure(figsize=(12, 6))
    results_df.plot(kind='bar', figsize=(12, 6))
    plt.title(f'Comparação entre Tamanhos do Estado Escondido ({best_rnn_type.upper()})')
    plt.ylabel('Valor')
    plt.xlabel('Tamanho do Estado Escondido')
    plt.xticks(rotation=0)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{best_rnn_type}_hidden_size_comparison.png'))
    plt.show()
    
    return results_df

def main():
    config = ExperimentConfig()
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    
    # Timestamp para o diretório de experimentos
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.RESULTS_DIR = f'experiment_results_{timestamp}'
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    # Carregar dados
    train_data = PoseDataset('data/splits/train', seq_length=config.SEQUENCE_LENGTH)
    val_data = PoseDataset('data/splits/val', seq_length=config.SEQUENCE_LENGTH)
    test_data = PoseDataset('data/splits/test', seq_length=config.SEQUENCE_LENGTH)
    
    # Criar DataLoaders
    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.BATCH_SIZE)
    test_loader = DataLoader(test_data, batch_size=config.BATCH_SIZE)
    
    # Verificar tamanho da entrada
    sample_input = train_data[0][0]
    input_size = sample_input.shape[-1]  # (seq_len, input_size)
    
    print(f"Tamanho da entrada: {input_size}")
    print(f"Total de amostras: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # Configurações do experimento em JSON
    with open(os.path.join(config.RESULTS_DIR, 'config.json'), 'w') as f:
        config_dict = {k: v for k, v in vars(config).items() if not k.startswith('_') and k != 'DEVICE'}
        config_dict['DEVICE'] = str(config.DEVICE)
        json.dump(config_dict, f, indent=4)
    
    # Experimento 1: Comparar tipos de RNN
    print("\n=== EXPERIMENTO 1: COMPARAÇÃO DE TIPOS DE RNN ===")
    rnn_results = run_type_experiment(input_size, train_loader, val_loader, test_loader, config)
    
    # Determinar o melhor tipo de RNN baseado na acurácia
    best_rnn_type = rnn_results['test_accuracy'].idxmax()
    print(f"\nMelhor tipo de RNN: {best_rnn_type.upper()}")
    
    # Experimento 2: Comparar tamanhos de hidden state para o melhor tipo de RNN
    print(f"\n=== EXPERIMENTO 2: COMPARAÇÃO DE TAMANHOS DE HIDDEN STATE PARA {best_rnn_type.upper()} ===")
    hidden_results = run_hidden_size_experiment(input_size, train_loader, val_loader, test_loader, config, best_rnn_type)
    
    # Determinar o melhor tamanho de hidden state
    best_hidden_size = hidden_results['test_accuracy'].idxmax()
    print(f"\nMelhor tamanho de hidden state: {best_hidden_size}")
    
    # Salvar resultados finais
    with open(os.path.join(config.RESULTS_DIR, 'final_results.txt'), 'w') as f:
        f.write(f"Resultados dos Experimentos - {timestamp}\n")
        f.write(f"===================================\n\n")
        f.write(f"Experimento 1: Comparação de Tipos de RNN\n")
        f.write(f"Melhor tipo de RNN: {best_rnn_type.upper()}\n\n")
        f.write(f"Experimento 2: Comparação de Tamanhos de Hidden State\n")
        f.write(f"Melhor tamanho de hidden state: {best_hidden_size}\n\n")
        f.write(f"Configuração recomendada:\n")
        f.write(f"- Tipo de RNN: {best_rnn_type.upper()}\n")
        f.write(f"- Tamanho do hidden state: {best_hidden_size}\n")
    
    print(f"\nExperimentos concluídos! Resultados salvos em {config.RESULTS_DIR}")

if __name__ == '__main__':
    main()