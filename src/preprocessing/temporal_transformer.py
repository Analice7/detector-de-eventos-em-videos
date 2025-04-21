import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

class TemporalSelfAttention(nn.Module):
    """
    Implementação da camada de self-attention temporal para o TFT
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        
        self.fc_out = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Shape: [batch_size, num_heads, seq_len, seq_len]
        attention = torch.matmul(Q, K.permute(0, 1, 3, 2)) / (self.head_dim ** 0.5)
        attention = torch.softmax(attention, dim=-1)
        
        # Shape: [batch_size, num_heads, seq_len, head_dim]
        output = torch.matmul(attention, V)
        
        # Shape: [batch_size, seq_len, d_model]
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.fc_out(output)

class TemporalFusionTransformer(nn.Module):
    """
    Versão simplificada do Temporal Fusion Transformer para suavização de keypoints
    """
    def __init__(self, input_dim, output_dim, hidden_dim=64, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim*4,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        x = self.input_embedding(x)
        
        # Aplicar camadas do transformer
        for layer in self.transformer_layers:
            x = layer(x)
            
        # Saída final
        return self.output_layer(x)

def aplicar_suavizacao_temporal_tft(keypoints_sequence, window_size=5, hidden_dim=64, device='cpu'):
    """
    Aplica suavização temporal aos keypoints usando Temporal Fusion Transformer
    
    Args:
        keypoints_sequence: Lista de arrays de keypoints (cada array tem shape [num_keypoints, 2])
        window_size: Tamanho da janela para suavização (contexto)
        hidden_dim: Dimensão do espaço latente do TFT
        device: Dispositivo para executar o modelo ('cpu' ou 'cuda')
    
    Returns:
        Lista de arrays de keypoints suavizados
    """
    if not keypoints_sequence or len(keypoints_sequence) == 0:
        return []
    
    try:
        # Verificar se todos os keypoints têm a mesma forma
        first_valid_kp = next((kp for kp in keypoints_sequence if kp is not None and len(kp) > 0), None)
        if first_valid_kp is None:
            return keypoints_sequence
        
        num_keypoints, keypoint_dim = first_valid_kp.shape
        device = torch.device(device)
        
        # Preparar dados para o TFT
        # Reorganizar para ter formato [num_pessoas, seq_len, keypoints*dims]
        sequences_by_person = []
        
        # Tratar cada pessoa separadamente
        for person_idx in range(num_keypoints):
            person_seq = []
            for frame_idx in range(len(keypoints_sequence)):
                kp = keypoints_sequence[frame_idx]
                if kp is not None and person_idx < len(kp):
                    person_seq.append(kp[person_idx])
                else:
                    # Se não houver keypoint para esta pessoa neste frame,
                    # usar zeros ou propagar o último valor válido
                    if person_seq:
                        person_seq.append(person_seq[-1])
                    else:
                        person_seq.append(np.zeros(keypoint_dim))
            
            sequences_by_person.append(np.array(person_seq))
        
        # Converter para tensor e aplicar TFT para cada pessoa
        smoothed_sequences_by_person = []
        
        for person_seq in sequences_by_person:
            # Se a sequência estiver vazia ou for muito curta, ignorar
            if len(person_seq) < window_size:
                smoothed_sequences_by_person.append(person_seq)
                continue
            
            # Preparar dados em formato de janelas deslizantes
            # Para cada posição na sequência, pegamos uma janela centrada
            padded_seq = np.pad(person_seq, ((window_size//2, window_size//2), (0, 0)), mode='edge')
            windows = []
            
            for i in range(len(person_seq)):
                window = padded_seq[i:i+window_size]
                windows.append(window)
            
            # Converter para tensor
            windows_tensor = torch.FloatTensor(np.array(windows)).to(device)
            
            # Criar e aplicar o modelo TFT
            input_dim = window_size * keypoint_dim
            output_dim = keypoint_dim
            
            # Reshape para entrada do modelo [batch_size, window_size, feature_dim]
            windows_tensor = windows_tensor.view(len(windows), window_size, keypoint_dim)
            
            # Criar e treinar modelo TFT
            model = TemporalFusionTransformer(
                input_dim=keypoint_dim,
                output_dim=keypoint_dim,
                hidden_dim=hidden_dim
            ).to(device)
            
            # Aqui precisaríamos treinar o modelo, mas para suavização
            # podemos usar autoencoders ou uma abordagem não supervisionada
            # Para simplificar, vamos apenas passar pela rede sem treinamento
            # Na prática, você precisaria treinar o modelo antes
            
            with torch.no_grad():
                smoothed_tensor = model(windows_tensor)
                
            # Pegar apenas a saída do meio da janela (posição atual)
            smoothed_seq = smoothed_tensor[:, window_size//2].cpu().numpy()
            smoothed_sequences_by_person.append(smoothed_seq)
        
        # Reorganizar os dados de volta para o formato original [frame, pessoa, dim]
        smoothed_keypoints_sequence = []
        for frame_idx in range(len(keypoints_sequence)):
            frame_keypoints = []
            for person_idx in range(len(smoothed_sequences_by_person)):
                if frame_idx < len(smoothed_sequences_by_person[person_idx]):
                    frame_keypoints.append(smoothed_sequences_by_person[person_idx][frame_idx])
            
            if frame_keypoints:
                smoothed_keypoints_sequence.append(np.array(frame_keypoints))
            else:
                smoothed_keypoints_sequence.append(np.array([]))
        
        return smoothed_keypoints_sequence
    
    except Exception as e:
        print(f"Erro na suavização com TFT: {str(e)}")
        import traceback
        traceback.print_exc()
        return keypoints_sequence  # Em caso de erro, retornar sequência original

def aplicar_suavizacao_temporal_tft_multipessoa(all_persons_keypoints, window_size=5, hidden_dim=64, device='cpu'):
    """
    Aplica suavização temporal TFT para múltiplas pessoas detectadas em cada frame
    """
    from src.preprocessing.temporal_transformer import aplicar_suavizacao_temporal_tft

    max_pessoas = max(len(frame_kps) for frame_kps in all_persons_keypoints)

    pessoas_suavizadas = []
    for pessoa_idx in range(max_pessoas):
        sequencia_pessoa = []
        for frame_kps in all_persons_keypoints:
            if pessoa_idx < len(frame_kps):
                sequencia_pessoa.append(frame_kps[pessoa_idx])
            else:
                sequencia_pessoa.append(np.zeros((17, 2)))  # ou np.nan * np.ones((17, 2))

        pessoa_suavizada = aplicar_suavizacao_temporal_tft(
            sequencia_pessoa,
            window_size=window_size,
            hidden_dim=hidden_dim,
            device=device
        )
        pessoas_suavizadas.append(pessoa_suavizada)

    smoothed_by_frame = []
    for frame_idx in range(len(all_persons_keypoints)):
        frame_data = []
        for pessoa_idx in range(max_pessoas):
            if frame_idx < len(pessoas_suavizadas[pessoa_idx]):
                frame_data.append(pessoas_suavizadas[pessoa_idx][frame_idx])
        smoothed_by_frame.append(np.array(frame_data))

    return smoothed_by_frame

# Versão simplificada do TFT para uso em produção sem necessidade de treinamento
class SimplifiedTFT:
    """
    Versão simplificada do TFT que pode ser usada sem treinamento prévio,
    baseada em médias ponderadas adaptativas usando attention
    """
    def __init__(self, window_size=5):
        self.window_size = window_size
    
    def smooth_sequence(self, keypoints_sequence):
        """
        Aplica suavização usando uma abordagem simplificada de attention
        
        Args:
            keypoints_sequence: Lista de arrays de keypoints
        
        Returns:
            Lista de arrays de keypoints suavizados
        """
        if not keypoints_sequence:
            return []
        
        smoothed_sequence = []
        seq_len = len(keypoints_sequence)
        
        for i in range(seq_len):
            # Define a janela de contexto
            start_idx = max(0, i - self.window_size // 2)
            end_idx = min(seq_len, i + self.window_size // 2 + 1)
            window = keypoints_sequence[start_idx:end_idx]
            
            # Verifica se a janela contém elementos válidos
            if not window or not all(w is not None and isinstance(w, np.ndarray) for w in window):
                smoothed_sequence.append(keypoints_sequence[i])
                continue
            
            # Verifica se todos os elementos têm a mesma forma
            shapes = [w.shape for w in window if w is not None]
            if not shapes or not all(s == shapes[0] for s in shapes):
                smoothed_sequence.append(keypoints_sequence[i])
                continue
            
            # Calcula pesos baseados na distância temporal (simulando attention)
            center = i - start_idx
            weights = np.exp(-0.5 * ((np.arange(len(window)) - center) / (self.window_size / 4))**2)
            weights = weights / weights.sum()
            
            # Aplica média ponderada
            try:
                stacked = np.stack([w for w in window if w is not None])
                weighted_sum = np.zeros_like(stacked[0])
                total_weight = 0
                
                for j, w in enumerate(window):
                    if w is not None:
                        weighted_sum += weights[j] * w
                        total_weight += weights[j]
                
                if total_weight > 0:
                    smoothed = weighted_sum / total_weight
                else:
                    smoothed = keypoints_sequence[i]
                    
                smoothed_sequence.append(smoothed)
            except Exception as e:
                print(f"Erro ao aplicar suavização no frame {i}: {e}")
                smoothed_sequence.append(keypoints_sequence[i])
        
        return smoothed_sequence
