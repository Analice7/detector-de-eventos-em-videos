import numpy as np
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import random
import shutil
import os
from .representation import calculate_velocity, calculate_acceleration

FEATURE_SIZE = 100  # Tamanho fixo para todas as features

def load_keypoints_sequence(keypoints_dir):
    """Carrega a sequência de keypoints dos arquivos .npy para todas as pessoas"""
    keypoints_dir = Path(keypoints_dir)
    files = sorted(list(keypoints_dir.glob("frame_*.npy")))
    
    # Dicionário para armazenar sequências de cada pessoa
    person_sequences = {}
    
    for file in files:
        try:
            keypoints = np.load(file)
            
            # Se for formato 3D (múltiplas pessoas)
            if len(keypoints.shape) == 3 and keypoints.shape[0] > 0:
                for person_idx in range(keypoints.shape[0]):
                    if person_idx not in person_sequences:
                        person_sequences[person_idx] = []
                    person_sequences[person_idx].append(keypoints[person_idx])
            
            # Se for formato 2D (uma única pessoa)
            elif len(keypoints.shape) == 2 and keypoints.shape[0] > 0:
                if 0 not in person_sequences:
                    person_sequences[0] = []
                person_sequences[0].append(keypoints)
                
        except Exception as e:
            print(f"Erro ao carregar {file}: {e}")
    
    # Verificar se há alguma sequência válida
    if not person_sequences:
        print(f"Nenhum keypoint válido encontrado em {keypoints_dir}")
        return []

    for person_id in person_sequences:
        person_sequences[person_id] = np.array(person_sequences[person_id])
    
    return person_sequences

def calculate_angle(p1, p2, p3):
    """Calcula o ângulo entre três pontos (p1-p2-p3)"""
    v1 = p1 - p2
    v2 = p3 - p2
    
    # Produto escalar e normalização
    dot = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # Evitar divisão por zero
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    
    # Calcular ângulo
    cos_angle = dot / (norm_v1 * norm_v2)
    cos_angle = np.clip(cos_angle, -1, 1)
    angle = np.degrees(np.arccos(cos_angle))
    
    return angle

def extract_features(keypoints_sequence, feature_type="all", normalize=True):
    """
    Extrai características dos keypoints.
    
    Args:
        keypoints_sequence: Dicionário de arrays de keypoints por pessoa
        feature_type: Tipo de característica ('position', 'angle', 'velocity', 'acceleration', 'all')
        normalize: Se True, normaliza as características
    
    Returns:
        Dicionário com as características extraídas
    """
    # Verificar se keypoints_sequence é um dicionário ou um array
    if isinstance(keypoints_sequence, dict):
        # Se for um dicionário, vamos usar apenas a primeira pessoa (pessoa 0)
        if 0 in keypoints_sequence:
            keypoints_array = keypoints_sequence[0]
        else:
            # Use a primeira chave disponível no dicionário
            first_key = next(iter(keypoints_sequence))
            keypoints_array = keypoints_sequence[first_key]
    else:
        # Se já for um array, use-o diretamente
        keypoints_array = keypoints_sequence
    
    if keypoints_array.size == 0:
        return {}
    
    features = {}
    n_frames = keypoints_array.shape[0]
    
    # Verificar forma para determinar quantos pontos temos
    n_keypoints = keypoints_array.shape[1] if len(keypoints_array.shape) > 1 else 0
    
    # 1. Posições absolutas
    if feature_type in ["position", "all"]:
        for i in range(n_keypoints):
            try:
                # Verificar se o keypoint existe em todos os frames
                if i < keypoints_array.shape[1]:
                    # Extrair coordenadas x e y
                    x_coords = keypoints_array[:, i, 0] if keypoints_array.shape[2] > 0 else np.zeros(n_frames)
                    y_coords = keypoints_array[:, i, 1] if keypoints_array.shape[2] > 1 else np.zeros(n_frames)
                    
                    features[f"kp{i}_x"] = x_coords
                    features[f"kp{i}_y"] = y_coords
                
            except Exception as e:
                print(f"Erro ao processar keypoint {i}: {e}")
                # Pular este keypoint
                continue
    
    # 2. Ângulos importantes
    if feature_type in ["angle", "all"]:
        # Definição dos ângulos a calcular: (p1, ponto_central, p2, nome)
        angle_configs = [
            (5, 7, 9, "left_elbow"),     # ombro esquerdo, cotovelo esquerdo, pulso esquerdo
            (6, 8, 10, "right_elbow"),   # ombro direito, cotovelo direito, pulso direito
            (12, 14, 16, "left_knee"),   # quadril esquerdo, joelho esquerdo, tornozelo esquerdo
            (13, 15, 17, "right_knee"),  # quadril direito, joelho direito, tornozelo direito
        ]
        
        for p1_idx, p2_idx, p3_idx, name in angle_configs:
            try:
                angles = np.zeros(n_frames)
                
                # Verificar se todos os índices estão dentro dos limites
                max_idx = max(p1_idx, p2_idx, p3_idx)
                if max_idx < keypoints_array.shape[1] and keypoints_array.shape[2] >= 2:
                    for i in range(n_frames):
                        try:
                            p1 = keypoints_array[i, p1_idx, :2]  # Usar apenas x,y
                            p2 = keypoints_array[i, p2_idx, :2]
                            p3 = keypoints_array[i, p3_idx, :2]
                            angles[i] = calculate_angle(p1, p2, p3)
                        except Exception:
                            # Se houver erro no cálculo, usar valor anterior ou zero
                            angles[i] = angles[i-1] if i > 0 else 0
                
                    features[f"angle_{name}"] = angles
                
            except Exception as e:
                print(f"Erro ao calcular ângulo {name}: {e}")
                continue
    
    # 3. Velocidades (usando a função melhorada)
    if feature_type in ["velocity", "all"]:
        try:
            # Calcular velocidades para todos os keypoints de uma vez
            velocities = calculate_velocity(keypoints_array)
            
            # Lidar com a diferença de tamanho (velocidade tem n_frames-1)
            vel_pad = np.zeros((1, velocities.shape[1], velocities.shape[2]))
            velocities_padded = np.concatenate([vel_pad, velocities], axis=0)  # Adiciona um frame zero no início
            
            # Extrair componentes x e y da velocidade para cada keypoint
            for i in range(n_keypoints):
                if i < velocities_padded.shape[1]:
                    features[f"kp{i}_x_vel"] = velocities_padded[:, i, 0]
                    features[f"kp{i}_y_vel"] = velocities_padded[:, i, 1]
                    
                    # Calcular magnitude da velocidade
                    magnitude = np.sqrt(velocities_padded[:, i, 0]**2 + velocities_padded[:, i, 1]**2)
                    features[f"kp{i}_vel_magnitude"] = magnitude
                
        except Exception as e:
            print(f"Erro ao calcular velocidades: {e}")
    
    # 4. Acelerações
    if feature_type in ["acceleration", "all"]:
        try:
            # Calcular acelerações para todos os keypoints de uma vez
            accelerations = calculate_acceleration(keypoints_array)
            
            # Lidar com a diferença de tamanho (aceleração tem n_frames-2)
            acc_pad = np.zeros((2, accelerations.shape[1], accelerations.shape[2]))
            accelerations_padded = np.concatenate([acc_pad, accelerations], axis=0)  # Adiciona dois frames zero no início
            
            # Extrair componentes x e y da aceleração para cada keypoint
            for i in range(n_keypoints):
                if i < accelerations_padded.shape[1]:
                    features[f"kp{i}_x_acc"] = accelerations_padded[:, i, 0]
                    features[f"kp{i}_y_acc"] = accelerations_padded[:, i, 1]
                    
                    # Calcular magnitude da aceleração
                    magnitude = np.sqrt(accelerations_padded[:, i, 0]**2 + accelerations_padded[:, i, 1]**2)
                    features[f"kp{i}_acc_magnitude"] = magnitude
                
        except Exception as e:
            print(f"Erro ao calcular acelerações: {e}")
    
    # Normalização
    if normalize and features:
        for key in list(features.keys()):  # Usar list() para evitar erro de modificação durante iteração
            try:
                values = features[key]
                min_val = np.min(values)
                max_val = np.max(values)
                
                if max_val > min_val:
                    features[key] = (values - min_val) / (max_val - min_val)
            except Exception as e:
                print(f"Erro ao normalizar {key}: {e}")
    
    return features

def process_video_keypoints(keypoints_dir, output_dir=None, feature_type="all", normalize=True, window_size=30):
    """
    Processa os keypoints de um vídeo e extrai características.
    
    Args:
        keypoints_dir: Diretório com os arquivos de keypoints
        output_dir: Diretório para salvar as características (opcional)
        feature_type: Tipo de característica a extrair
        normalize: Se True, normaliza as características
        
    Returns:
        Dicionário com as características extraídas
    """
    try:
        # Carregar keypoints
        keypoints_sequence = load_keypoints_sequence(keypoints_dir)
        
        if not keypoints_sequence:
            print(f"Nenhum keypoint válido encontrado em {keypoints_dir}")
            return {}
        
        # Extrair características
        features_dict = extract_features(keypoints_sequence, feature_type, normalize)
        
        if not features_dict:
            print(f"Nenhuma característica extraída de {keypoints_dir}")
            return {}
        
        # Converter o dicionário em uma sequência de valores para treinamento
        # Cada frame será representado por um vetor contendo todas as características
        n_frames = len(next(iter(features_dict.values())))
        feature_keys = sorted(features_dict.keys())  # Ordena as chaves para consistência
        
        # Inicializar matriz de sequência (n_frames x n_features)
        # VETORES DE CARACTERÍSTICAS
        feature_sequence = np.zeros((n_frames, len(feature_keys)))
        
        # Preencher a matriz
        for i, key in enumerate(feature_keys):
            feature_sequence[:, i] = features_dict[key]

        windows = split_into_windows(feature_sequence, window_size)

        # Salvar se o diretório de saída for especificado
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Salvar o dicionário original para referência/visualização
            np.save(output_dir / "features_dict.npy", features_dict)
            
            # Salvar a sequência para treinamento
            np.save(output_dir / "features.npy", feature_sequence)

            # Salva em janelas
            np.save(output_dir / "windows.npy", windows)
            
            # Tentar gerar visualização se houver características
            try:
                plt.figure(figsize=(12, 6))
                
                # Selecionar até 5 características (se disponíveis)
                keys_to_plot = list(features_dict.keys())[:min(5, len(features_dict))]
                
                if keys_to_plot:
                    for key in keys_to_plot:
                        plt.plot(features_dict[key], label=key)
                    
                    plt.title("Exemplo de Características Extraídas")
                    plt.xlabel("Frame")
                    plt.ylabel("Valor Normalizado")
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(output_dir / "features_preview.png")
                
                plt.close()
            except Exception as e:
                print(f"Erro ao gerar visualização: {e}")
        
        return features_dict
    
    except Exception as e:
        print(f"Erro ao processar vídeo {keypoints_dir}: {e}")
        return {}

def process_dataset(process_type, base_keypoints_dir, output_base_dir, feature_type="all", normalize=True, window_size=16):
    """
    Processa todos os vídeos no conjunto de dados.
    
    Args:
        base_keypoints_dir: Diretório base com os keypoints
        output_base_dir: Diretório base para salvar as características
        feature_type: Tipo de característica a extrair
        normalize: Se True, normaliza as características
    """
    base_keypoints_dir = Path(base_keypoints_dir)
    output_base_dir = Path(output_base_dir)
    
    # Criar diretório de saída
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Verificar se o diretório smoothed existe
    if not base_keypoints_dir.exists():
        print(f"Diretório de keypoints suavizados não encontrado: {base_keypoints_dir}")
        return
    
    # Para cada classe dentro dos diretórios
    for class_dir in base_keypoints_dir.iterdir():
        if not class_dir.is_dir():
            continue
            
        class_name = class_dir.name  # assault ou normal
        class_output_dir = output_base_dir / class_name
        class_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Processando classe: {class_name}")
        
        # Para cada vídeo na classe
        for video_dir in class_dir.iterdir():
            if not video_dir.is_dir():
                continue
                
            video_name = video_dir.name
            video_output_dir = class_output_dir / video_name
            
            print(f"  Extraindo características: {video_name}")
            
            process_video_keypoints(
                keypoints_dir=video_dir,  
                output_dir=str(video_output_dir),
                feature_type=feature_type,
                normalize=normalize,
                window_size=window_size
            )
    
    split(
        base_dir='data/processed/sequences/'+process_type,
        output_dir='data/splits/'+process_type
    )

# Desabilitar mensagens de aviso do matplotlib para evitar problemas de QT
warnings.filterwarnings("ignore")

# Definir backend não-interativo para matplotlib
matplotlib.use('Agg')

def split(base_dir, output_dir, ratios=(0.7, 0.1, 0.2), seed=42):
    """
    Divide os vídeos processados em pastas de treino, validação e teste.
    
    Args:
        base_dir: Pasta com as classes (assault/normal) e vídeos processados
        output_dir: Onde criar as pastas train/val/test
        ratios: Proporções para (treino, validação, teste)
        seed: Semente para reprodutibilidade
    """
    random.seed(seed)
    
    # Cria pastas de destino
    for split in ['train', 'val', 'test']:
        for classe in ['assault', 'normal']:
            os.makedirs(f'{output_dir}/{split}/{classe}', exist_ok=True)
    
    # Para cada classe
    for classe in ['assault', 'normal']:
        videos = os.listdir(f'{base_dir}/{classe}')
        random.shuffle(videos)
        
        n = len(videos)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        
        # Divide
        train = videos[:n_train]
        val = videos[n_train:n_train+n_val]
        test = videos[n_train+n_val:]
        
        # Copia arquivos
        for video in train:
            src = f'{base_dir}/{classe}/{video}'
            dst = f'{output_dir}/train/{classe}/{video}'
            shutil.copytree(src, dst)
        
        for video in val:
            src = f'{base_dir}/{classe}/{video}'
            dst = f'{output_dir}/val/{classe}/{video}'
            shutil.copytree(src, dst)
        
        for video in test:
            src = f'{base_dir}/{classe}/{video}'
            dst = f'{output_dir}/test/{classe}/{video}'
            shutil.copytree(src, dst)

def split_into_windows(sequence, window_size, overlap=0.5):
    windows = []
    n_frames = sequence.shape[0]
    step = int(window_size * (1 - overlap))
    
    for i in range(0, n_frames - window_size + 1, step):
        windows.append(sequence[i:i+window_size])
    
    # Se não houve nenhuma janela completa, pega o que tiver
    if not windows and n_frames > 0:
        last_window = sequence[-window_size:] if n_frames >= window_size else sequence
        windows.append(last_window)
    
    return np.array(windows) if windows else np.zeros((1, window_size, sequence.shape[1]))

if __name__ == "__main__":
    # Configurações
    KEYPOINTS_DIR = "data/processed/keypoints/"  # Diretório base contém "smoothed" e "no_smoothed"
    OUTPUT_DIR = "data/processed/sequences/"
    
    print("Iniciando extração de características...")
    
    # Processar todo o conjunto de dados
    for process_type in ['no_smoothed', 'smoothed']:
        process_dataset(
            process_type,
            base_keypoints_dir=KEYPOINTS_DIR + process_type,
            output_base_dir=OUTPUT_DIR + process_type,
            feature_type="all",  # Extrair todos os tipos de características
            normalize=True,       # Normalizar as características
            window_size=64
        )

    print("Extração de características concluída!")