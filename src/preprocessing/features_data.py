import numpy as np
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import random
import shutil
import pandas as pd
import seaborn as sns
import os

FEATURE_SIZE = 100  # Tamanho fixo para todas as features

def load_keypoints_sequence(keypoints_dir):
    """Carrega a sequência de keypoints dos arquivos .npy"""
    keypoints_dir = Path(keypoints_dir)
    files = sorted(list(keypoints_dir.glob("frame_*.npy")))
    
    sequence = []
    for file in files:
        try:
            keypoints = np.load(file)
            
            # Verificar se o array tem formato válido
            if len(keypoints.shape) == 3 and keypoints.shape[0] > 0 and keypoints.shape[1] > 0:
                sequence.append(keypoints[0])  # Primeira pessoa detectada
            elif len(keypoints.shape) == 2 and keypoints.shape[0] > 0:
                sequence.append(keypoints)     # Array já formatado corretamente
            else:
                # Array vazio ou formato inválido, pular este frame
                continue
                
        except Exception as e:
            print(f"Erro ao carregar {file}: {e}")
    
    # Verificar se há keypoints válidos
    if not sequence:
        print(f"Nenhum keypoint válido encontrado em {keypoints_dir}")
        return []
    
    return sequence

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
        keypoints_sequence: Lista de arrays de keypoints
        feature_type: Tipo de característica ('position', 'angle', 'velocity', 'all')
        normalize: Se True, normaliza as características
    
    Returns:
        Dicionário com as características extraídas
    """
    if not keypoints_sequence:
        return {}
    
    features = {}
    n_frames = len(keypoints_sequence)
    
    # Verificar forma do primeiro keypoint para determinar quantos pontos temos
    n_keypoints = keypoints_sequence[0].shape[0]
    
    # 1. Posições absolutas
    if feature_type in ["position", "all"]:
        for i in range(n_keypoints):
            try:
                # Usar lista por compreensão com verificação de índice
                x_coords = []
                y_coords = []
                
                for keypoints in keypoints_sequence:
                    if i < keypoints.shape[0] and keypoints.shape[1] >= 2:
                        x_coords.append(keypoints[i, 0])
                        y_coords.append(keypoints[i, 1])
                    else:
                        # Se o keypoint não existir neste frame, usar valor anterior ou zero
                        x_val = x_coords[-1] if x_coords else 0
                        y_val = y_coords[-1] if y_coords else 0
                        x_coords.append(x_val)
                        y_coords.append(y_val)
                
                # Converter para arrays numpy
                x_coords = np.array(x_coords)
                y_coords = np.array(y_coords)
                
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
                
                for i, keypoints in enumerate(keypoints_sequence):
                    try:
                        # Verificar se todos os keypoints necessários estão presentes
                        if (max(p1_idx, p2_idx, p3_idx) < keypoints.shape[0] and
                            keypoints.shape[1] >= 2):
                            p1 = keypoints[p1_idx]
                            p2 = keypoints[p2_idx]
                            p3 = keypoints[p3_idx]
                            angles[i] = calculate_angle(p1, p2, p3)
                    except Exception:
                        # Se algum keypoint estiver ausente ou erro no cálculo, usar zero
                        angles[i] = angles[i-1] if i > 0 else 0
                
                features[f"angle_{name}"] = angles
                
            except Exception as e:
                print(f"Erro ao calcular ângulo {name}: {e}")
                continue
    
    # 3. Velocidades (derivadas de primeira ordem)
    if feature_type in ["velocity", "all"]:
        # Para cada posição, calcular a velocidade
        position_keys = [k for k in features.keys() if k.startswith("kp")]
        
        for key in position_keys:
            try:
                values = features[key]
                velocity = np.zeros_like(values)
                velocity[1:] = values[1:] - values[:-1]
                
                features[f"{key}_vel"] = velocity
            except Exception as e:
                print(f"Erro ao calcular velocidade para {key}: {e}")
                continue  
    
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
                # Manter o valor original em caso de erro
    
    return features

def process_video_keypoints(keypoints_dir, output_dir=None, feature_type="all", normalize=True, window_size = 30):
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

def process_dataset(base_keypoints_dir, output_base_dir, feature_type="all", normalize=True, window_size=16):
    base_keypoints_dir = Path(base_keypoints_dir)
    stats = []

    for class_dir in base_keypoints_dir.iterdir():
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        print(f"Processando classe: {class_name}")

        for video_dir in class_dir.iterdir():
            if not video_dir.is_dir():
                continue

            video_name = video_dir.name
            print(f"  Extraindo características: {video_name}")

            keypoints_sequence = load_keypoints_sequence(video_dir / "smoothed")
            if not keypoints_sequence:
                continue

            features_dict = extract_features(keypoints_sequence, feature_type, normalize)
            if not features_dict:
                continue

            n_frames = len(next(iter(features_dict.values())))
            feature_keys = sorted(features_dict.keys())
            feature_sequence = np.zeros((n_frames, len(feature_keys)))

            for i, key in enumerate(feature_keys):
                feature_sequence[:, i] = features_dict[key]

            windows = split_into_windows(feature_sequence, window_size)

            # Cálculo da variação média das janelas (como proxy de informação útil)
            if len(windows) > 0:
                for w in windows:
                    mean_std = np.mean(np.std(w, axis=0))  # média dos desvios padrão
                    stats.append({
                        "classe": class_name,
                        "video": video_name,
                        "feature_type": feature_type,
                        "window_size": window_size,
                        "variabilidade": mean_std
                    })

    return stats

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

def split_into_windows(sequence, window_size):
    # Processamento em janelas de tempo
    windows = []
    n_frames = sequence.shape[0]
    for i in range(0, n_frames - window_size + 1, window_size):
        windows.append(sequence[i:i+window_size])
    return np.array(windows)

if __name__ == "__main__":

    KEYPOINTS_DIR = "data/processed/keypoints"
    FIXED_WINDOW_SIZE = 64
    FEATURE_TYPES = ["position", "angle", "velocity", "all"]

    all_stats = []

    for FEATURE_TYPE in FEATURE_TYPES:
        print(f"===== AVALIANDO FEATURE: {FEATURE_TYPE} com WINDOW_SIZE = {FIXED_WINDOW_SIZE} =====")
        
        stats = process_dataset(
            base_keypoints_dir=KEYPOINTS_DIR,
            output_base_dir="",  # Não estamos salvando arquivos
            feature_type=FEATURE_TYPE,
            normalize=True,
            window_size=FIXED_WINDOW_SIZE
        )
        all_stats.extend(stats)

    df_stats = pd.DataFrame(all_stats)

    # Salvar os dados em CSV para análise posterior (opcional)
    df_stats.to_csv("eficacia_features.csv", index=False)

    # Plotar gráfico comparativo
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=df_stats, x="feature_type", y="variabilidade", hue="classe")
    plt.title(f"Eficácia dos vetores de características (window = {FIXED_WINDOW_SIZE})")
    plt.xlabel("Tipo de Feature")
    plt.ylabel("Variabilidade Média das Janelas")
    plt.legend(title="Classe")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("eficacia_comparacao_features.png")
    plt.close()
