import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os

# Configurações
class Config:
    SEED = 42
    BATCH_SIZE = 32
    HIDDEN_SIZE = 512
    NUM_LAYERS = 2
    DROPOUT = 0.3
    LEARNING_RATE = 0.001
    EPOCHS = 50
    MAX_SEQUENCE_LENGTH = 64    # Tamanho máximo de sequência
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset customizado para lidar com sequências variáveis e diferentes dimensões
class PoseDataset(Dataset):
    def __init__(self, features_dir, max_seq_length=Config.MAX_SEQUENCE_LENGTH, feature_dim=139):
        self.features = []
        self.lengths = []  # Armazenar comprimentos das sequências
        self.labels = []
        self.max_seq_length = max_seq_length
        self.feature_dim = feature_dim  # Dimensão padrão das features (mais comum)
        
        # Carrega features e labels
        for class_idx, class_name in enumerate(['normal', 'assault']):
            class_dir = Path(features_dir) / class_name
            for video_dir in class_dir.iterdir():
                if video_dir.is_dir():
                    feature_file = video_dir / 'features.npy'
                    if feature_file.exists():
                        try:
                            # Carregar a matriz de features (n_frames x n_features)
                            feature_data = np.load(feature_file, allow_pickle=True)
                            
                            # Verificar se os dados têm forma adequada
                            if len(feature_data.shape) == 2:
                                n_frames, n_features = feature_data.shape
                                
                                # Verificar se a dimensão de features é a esperada
                                if n_features != self.feature_dim:
                                    print(f"Dimensão de features diferente em {feature_file}: {n_features} (esperado {self.feature_dim})")
                                    
                                    if n_features < self.feature_dim:
                                        # Padding para alcançar a dimensão padrão
                                        padded_data = np.zeros((n_frames, self.feature_dim))
                                        padded_data[:, :n_features] = feature_data
                                        feature_data = padded_data
                                    else:
                                        # Truncar para a dimensão padrão
                                        feature_data = feature_data[:, :self.feature_dim]
                                
                                # Se a sequência for muito longa, dividi-la em partes
                                if n_frames > self.max_seq_length:
                                    # Sobreposição de 50%
                                    step = self.max_seq_length // 2
                                    for i in range(0, n_frames - self.max_seq_length + 1, step):
                                        seq = feature_data[i:i+self.max_seq_length]
                                        if len(seq) >= 8:  # Mínimo de 8 frames para ser válido
                                            self.features.append(seq)
                                            self.lengths.append(len(seq))
                                            self.labels.append(class_idx)
                                else:
                                    # Se for menor que o tamanho máximo, usar a sequência completa
                                    if n_frames >= 8:  # Mínimo de 8 frames
                                        self.features.append(feature_data)
                                        self.lengths.append(n_frames)
                                        self.labels.append(class_idx)
                            elif isinstance(feature_data, dict):
                                # Compatibilidade com o formato antigo (dicionário)
                                print(f"Formato antigo detectado em {feature_file}, convertendo...")
                                feature_keys = sorted(feature_data.keys())
                                n_frames = len(feature_data[feature_keys[0]])
                                
                                # Criar matriz de sequência
                                seq_matrix = np.zeros((n_frames, len(feature_keys)))
                                for i, key in enumerate(feature_keys):
                                    seq_matrix[:, i] = feature_data[key]
                                
                                # Ajustar dimensão das características se necessário
                                n_features = seq_matrix.shape[1]
                                if n_features != self.feature_dim:
                                    if n_features < self.feature_dim:
                                        padded_matrix = np.zeros((n_frames, self.feature_dim))
                                        padded_matrix[:, :n_features] = seq_matrix
                                        seq_matrix = padded_matrix
                                    else:
                                        seq_matrix = seq_matrix[:, :self.feature_dim]
                                
                                # Processar como acima
                                if n_frames > self.max_seq_length:
                                    step = self.max_seq_length // 2
                                    for i in range(0, n_frames - self.max_seq_length + 1, step):
                                        seq = seq_matrix[i:i+self.max_seq_length]
                                        if len(seq) >= 8:
                                            self.features.append(seq)
                                            self.lengths.append(len(seq))
                                            self.labels.append(class_idx)
                                else:
                                    if n_frames >= 8:
                                        self.features.append(seq_matrix)
                                        self.lengths.append(n_frames)
                                        self.labels.append(class_idx)
                            # Lidar com casos especiais: array 1D ou entrada inválida
                            elif len(feature_data.shape) == 1:
                                print(f"Formato inválido detectado em {feature_file}: {feature_data.shape}, pulando...")
                            else:
                                print(f"Formato desconhecido em {feature_file}: {feature_data.shape}, pulando...")
                        except Exception as e:
                            print(f"Erro ao carregar {feature_file}: {e}")
                            
        if len(self.features) == 0:
            print(f"AVISO: Nenhum dado válido encontrado em {features_dir}")
        else:
            print(f"Carregados {len(self.features)} exemplos com dimensão de feature {self.feature_dim}")

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Verifica e ajusta dimensão das features se necessário
        feature = self.features[idx]
        if feature.shape[-1] != self.feature_dim:
            if feature.shape[-1] < self.feature_dim:
                # Padding
                padded = np.zeros((feature.shape[0], self.feature_dim))
                padded[:, :feature.shape[1]] = feature
                feature = padded
            else:
                # Truncar
                feature = feature[:, :self.feature_dim]
                
        # Retorna a sequência ajustada
        return feature, self.labels[idx], self.lengths[idx]

# Função collate personalizada para processar lotes
def collate_fn(batch):
    # Verificar se há dados no batch
    if len(batch) == 0:
        return None, None, None
        
    # Ordenar o batch por comprimento decrescente (importante para pack_padded_sequence)
    batch.sort(key=lambda x: x[2], reverse=True)
    
    # Separar features, labels e comprimentos
    features = []
    labels = []
    lengths = []
    
    for feature, label, length in batch:
        features.append(torch.FloatTensor(feature))
        labels.append(label)
        lengths.append(length)
    
    # Garantir que todas as features tenham a mesma dimensão
    feature_dims = [f.shape[1] for f in features]
    if not all(d == feature_dims[0] for d in feature_dims):
        max_dim = max(feature_dims)
        for i, feat in enumerate(features):
            if feat.shape[1] < max_dim:
                padding = torch.zeros(feat.shape[0], max_dim - feat.shape[1])
                features[i] = torch.cat([feat, padding], dim=1)
    
    # Padding das sequências (comprimento)
    padded_features = nn.utils.rnn.pad_sequence(features, batch_first=True)
    
    return padded_features, torch.LongTensor(labels), lengths

# Modelo RNN modificado para lidar com sequências de comprimento variável
class PoseRNN(nn.Module):
    def __init__(self, input_size, rnn_type='lstm'):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        
        # Camada RNN
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=Config.HIDDEN_SIZE,
                num_layers=Config.NUM_LAYERS,
                batch_first=True,
                dropout=Config.DROPOUT if Config.NUM_LAYERS > 1 else 0
            )
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=Config.HIDDEN_SIZE,
                num_layers=Config.NUM_LAYERS,
                batch_first=True,
                dropout=Config.DROPOUT if Config.NUM_LAYERS > 1 else 0
            )
        else:  # RNN simples
            self.rnn = nn.RNN(
                input_size=input_size,
                hidden_size=Config.HIDDEN_SIZE,
                num_layers=Config.NUM_LAYERS,
                batch_first=True,
                dropout=Config.DROPOUT if Config.NUM_LAYERS > 1 else 0
            )
        
        # Camada de classificação
        self.fc = nn.Linear(Config.HIDDEN_SIZE, 2)  # 2 classes (normal, assault)
        self.dropout = nn.Dropout(Config.DROPOUT)

    def forward(self, x, lengths):
        # Pack padded sequences para processamento eficiente
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=True
        )
        
        # Processar com a RNN
        if self.rnn_type == 'lstm':
            packed_out, (hidden, _) = self.rnn(packed_x)
            last_hidden = hidden[-1]  # Pegar o último estado oculto
        else:
            packed_out, hidden = self.rnn(packed_x)
            last_hidden = hidden[-1]  # Para GRU/RNN simples
            
        # Aplicar dropout e classificação
        out = self.dropout(last_hidden)
        out = self.fc(out)
        
        return out

# Função de treino modificada
def train_model(model, train_loader, val_loader, optimizer, criterion):
    best_acc = 0
    train_losses = []
    val_losses = []
    val_accs = []
    
    for epoch in range(Config.EPOCHS):
        model.train()
        running_loss = 0
        batch_count = 0
        
        # Loop de treino
        for inputs, labels, lengths in train_loader:
            # Verificar se temos dados válidos
            if inputs is None or labels is None:
                continue
                
            inputs = inputs.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)
            
            optimizer.zero_grad()
            
            try:
                outputs = model(inputs, lengths)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                batch_count += 1
            except Exception as e:
                print(f"Erro durante o treinamento: {e}")
                print(f"Dimensões de entrada: {inputs.shape}, comprimentos: {lengths}")
                continue
        
        # Evitar divisão por zero
        if batch_count == 0:
            print("Aviso: Nenhum batch foi processado nesta época!")
            continue
            
        # Validação
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        train_loss = running_loss / batch_count
        
        # Salvar métricas
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Salvar melhor modelo
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f'best_{model.rnn_type}_model.pth')
        
        print(f'Epoch {epoch+1}/{Config.EPOCHS} | '
              f'Train Loss: {train_loss:.4f} | '
              f'Val Loss: {val_loss:.4f} | '
              f'Val Acc: {val_acc:.2%}')

    return train_losses, val_losses, val_accs

# Função de avaliação modificada
def evaluate(model, loader, criterion):
    model.eval()
    total = 0
    correct = 0
    running_loss = 0
    batch_count = 0
    
    with torch.no_grad():
        for inputs, labels, lengths in loader:
            # Verificar se temos dados válidos
            if inputs is None or labels is None:
                continue
                
            inputs = inputs.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)
            
            try:
                outputs = model(inputs, lengths)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                batch_count += 1
            except Exception as e:
                print(f"Erro durante a avaliação: {e}")
                continue
    
    # Evitar divisão por zero
    if batch_count == 0:
        return 0, 0
        
    return running_loss / batch_count, correct / total if total > 0 else 0

# Pipeline principal
def main():
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    
    # Encontrar a dimensão de feature mais comum nos dados
    # Para simplificar, usaremos um valor fixo com base no seu erro (139)
    feature_dim = 139
    
    # Para cada tipo de processamento
    for process_type in ['smoothed', 'no_smoothed']:
        print(f"\n========== Processando dados {process_type} ==========")
        
        # Criar diretório para resultados
        output_dir = f'data/models/{process_type}'
        os.makedirs(output_dir, exist_ok=True)
        
        # Carregar dados específicos para este tipo de processamento
        train_data = PoseDataset(f'data/splits/{process_type}/train', feature_dim=feature_dim)
        val_data = PoseDataset(f'data/splits/{process_type}/val', feature_dim=feature_dim)
        test_data = PoseDataset(f'data/splits/{process_type}/test', feature_dim=feature_dim)
        
        # Verificar se temos dados suficientes
        if len(train_data) == 0 or len(val_data) == 0 or len(test_data) == 0:
            print(f"Dados insuficientes para {process_type}, pulando...")
            continue
            
        print(f"Amostras de treino: {len(train_data)}")
        print(f"Amostras de validação: {len(val_data)}")
        print(f"Amostras de teste: {len(test_data)}")
        
        # Criar DataLoaders com collate_fn personalizada
        train_loader = DataLoader(
            train_data, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=True, 
            collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_data, 
            batch_size=Config.BATCH_SIZE, 
            collate_fn=collate_fn
        )
        test_loader = DataLoader(
            test_data, 
            batch_size=Config.BATCH_SIZE, 
            collate_fn=collate_fn
        )
        
        # Usar dimensão de feature padrão para o modelo
        input_size = feature_dim
        
        # Testar diferentes modelos RNN
        for rnn_type in ['lstm', 'gru', 'rnn']:
            print(f'\n=== Treinando {rnn_type.upper()} com dados {process_type} ===')
            
            # Criar modelo
            model = PoseRNN(input_size, rnn_type=rnn_type).to(Config.DEVICE)
            
            # Loss e otimizador
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
            
            try:
                # Treinar
                train_losses, val_losses, val_accs = train_model(
                    model, train_loader, val_loader, optimizer, criterion)
                
                # Salvar modelo com nome específico para o tipo de processamento
                model_path = f'{output_dir}/best_{rnn_type}_model.pth'
                torch.save(model.state_dict(), model_path)
                print(f"Modelo salvo em: {model_path}")
                
                # Plotar resultados
                plt.figure(figsize=(12, 4))
                plt.subplot(1, 2, 1)
                plt.plot(train_losses, label='Train')
                plt.plot(val_losses, label='Validation')
                plt.title(f'{rnn_type.upper()} Loss - {process_type}')
                plt.legend()
                
                plt.subplot(1, 2, 2)
                plt.plot(val_accs, label='Validation Accuracy')
                plt.title(f'{rnn_type.upper()} Accuracy - {process_type}')
                plt.legend()
                
                plt.tight_layout()
                plot_path = f'{output_dir}/{rnn_type}_metrics.png'
                plt.savefig(plot_path)
                plt.close()
                print(f"Gráfico salvo em: {plot_path}")
                
                # Avaliar no teste
                print(f'\nAvaliação do {rnn_type.upper()} no conjunto de teste ({process_type}):')
                test_loss, test_acc = evaluate(model, test_loader, criterion)
                print(f'Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2%}')
                
                # Salvar relatório de classificação
                all_labels = []
                all_preds = []
                
                model.eval()
                with torch.no_grad():
                    for inputs, labels, lengths in test_loader:
                        if inputs is None or labels is None:
                            continue
                            
                        inputs = inputs.to(Config.DEVICE)
                        labels = labels.to(Config.DEVICE)
                        
                        try:
                            outputs = model(inputs, lengths)
                            _, predicted = torch.max(outputs.data, 1)
                            
                            all_labels.extend(labels.cpu().numpy())
                            all_preds.extend(predicted.cpu().numpy())
                        except Exception as e:
                            print(f"Erro durante o teste: {e}")
                            continue
                
                # Verificar se temos previsões suficientes
                if len(all_preds) == 0:
                    print("Aviso: Nenhuma previsão válida para gerar o relatório de classificação")
                    continue
                
                # Salvar métricas detalhadas
                report = classification_report(
                    np.array(all_labels), 
                    np.array(all_preds), 
                    target_names=['normal', 'assault'],
                    output_dict=True
                )
                
                with open(f'{output_dir}/{rnn_type}_report.txt', 'w') as f:
                    f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
                    f.write("Classification Report:\n")
                    f.write(classification_report(
                        np.array(all_labels), 
                        np.array(all_preds), 
                        target_names=['normal', 'assault']
                    ))
                
                # Salvar métricas como numpy para análise posterior
                metrics = {
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'val_accs': val_accs,
                    'test_acc': test_acc,
                    'test_loss': test_loss,
                    'report': report
                }
                np.save(f'{output_dir}/{rnn_type}_metrics.npy', metrics)
            
            except Exception as e:
                print(f"Erro ao treinar o modelo {rnn_type}: {e}")

if __name__ == '__main__':
    main()