import numpy as np
import cv2
from pathlib import Path
import pandas as pd
from ultralytics import YOLO
from ..metrics.keypoints import (
    gerar_visualizacoes_metricas,
    gerar_relatorio_metricas,
    calcular_metricas_avancadas,
    calcular_estabilidade_anatomica
)
import sys
from src.preprocessing.dataProcesser import extrair_frames

def aplicar_suavizacao_temporal(keypoints_sequence, window_size=5):
    """Aplica suavização temporal aos keypoints usando média móvel"""
    smoothed_sequence = []
    seq_len = len(keypoints_sequence)
    
    for i in range(seq_len):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(seq_len, i + window_size // 2 + 1)
        window = keypoints_sequence[start_idx:end_idx]
        
        # Verificar se todos os elementos em window têm a mesma forma
        if window and all(w is not None and isinstance(w, np.ndarray) and w.shape == window[0].shape for w in window):
            smoothed = np.mean(window, axis=0)
        else:
            smoothed = keypoints_sequence[i] if keypoints_sequence[i] is not None else np.array([])
        smoothed_sequence.append(smoothed)
    return smoothed_sequence

def draw_keypoints(frame, keypoints, pose_connections):
    """Desenha keypoints e conexões no frame"""
    if keypoints is None or len(keypoints) == 0:
        return frame
    
    frame_copy = frame.copy()
    h, w = frame.shape[:2]
    
    # Desenha conexões
    for i, j in pose_connections:
        if i < len(keypoints) and j < len(keypoints):
            # Verifica se os keypoints existem e têm confiança suficiente
            if (not np.any(np.isnan(keypoints[i])) and 
                not np.any(np.isnan(keypoints[j]))):
                
                pt1 = (int(keypoints[i][0] * w), int(keypoints[i][1] * h))
                pt2 = (int(keypoints[j][0] * w), int(keypoints[j][1] * h))
                
                # Verifica se os pontos estão dentro da imagem
                if (0 <= pt1[0] < w and 0 <= pt1[1] < h and 
                    0 <= pt2[0] < w and 0 <= pt2[1] < h):
                    cv2.line(frame_copy, pt1, pt2, (0, 255, 0), 2)
    
    # Desenha pontos
    for i, kp in enumerate(keypoints):
        if not np.any(np.isnan(kp)):
            x, y = int(kp[0] * w), int(kp[1] * h)
            if 0 <= x < w and 0 <= y < h:  # Verifica se está dentro da imagem
                cv2.circle(frame_copy, (x, y), 5, (0, 0, 255), -1)
    
    return frame_copy

def extract_keypoints_extended(frames, output_dir, apply_smoothing=True, window_size=5, save_raw=True):
    """Processa os frames e extrai keypoints"""
    try:
        model = YOLO('yolov8n-pose.pt')
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        raw_dir = output_dir / "raw"
        vis_dir = output_dir / "visualizations"
        smoothed_dir = output_dir / "smoothed"
        
        raw_dir.mkdir(exist_ok=True)
        vis_dir.mkdir(exist_ok=True)
        if apply_smoothing:
            smoothed_dir.mkdir(exist_ok=True)
            
        # Definição das conexões COCO
        pose_connections = [(0,1), (0,2), (1,3), (2,4), (5,6), (5,7), (7,9), (6,8), (8,10), 
                            (5,11), (6,12), (11,13), (12,14), (13,15), (14,16), (11,12)]
        
        all_keypoints = []
        all_persons_keypoints = []  # Lista para armazenar keypoints de todas as pessoas
        confidence_scores = []
        persons_count = []  # Lista para armazenar o número de pessoas em cada frame
        
        print("Processando frames...")
        for idx, frame in enumerate(frames):
            try:
                # Validação do frame
                if frame is None or frame.size == 0:
                    print(f"Frame {idx}: Vazio/corrompido - pulando")
                    all_keypoints.append(None)
                    all_persons_keypoints.append([])
                    confidence_scores.append(None)
                    persons_count.append(0)
                    continue
                
                # Conversão para formato compatível
                if len(frame.shape) == 2:  # Grayscale
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif frame.shape[2] == 4:  # Com canal alpha
                    frame = frame[:, :, :3]
                
                # Detecção
                results = model(frame, verbose=False)
                if not results or len(results) == 0:
                    all_keypoints.append(None)
                    all_persons_keypoints.append([])
                    confidence_scores.append(None)
                    persons_count.append(0)
                    continue
                
                # Extração de keypoints
                kps_data = results[0].keypoints
                if kps_data is None or kps_data.xyn is None or len(kps_data.xyn) == 0:
                    print(f"Frame {idx}: Nenhum keypoint detectado")
                    all_keypoints.append(None)
                    all_persons_keypoints.append([])
                    confidence_scores.append(None)
                    persons_count.append(0)
                    continue
                
                # Extrai keypoints de todas as pessoas
                frame_keypoints = kps_data.xyn.cpu().numpy()  # Todos os keypoints de todas as pessoas
                frame_conf = kps_data.conf.cpu().numpy() if hasattr(kps_data, 'conf') and kps_data.conf is not None else None
                
                # Salva visualização em PNG com todas as pessoas
                vis_frame = frame.copy()
                for person_kpts in frame_keypoints:
                    vis_frame = draw_keypoints(vis_frame, person_kpts, pose_connections)
                cv2.imwrite(str(vis_dir / f"frame_{idx:05d}.png"), vis_frame)
                
                # Salva keypoints em numpy (opcional)
                if save_raw:
                    np.save(raw_dir / f"frame_{idx:05d}.npy", frame_keypoints)
                
                # Armazena keypoints
                if len(frame_keypoints) > 0:
                    # Armazena keypoints da primeira pessoa para análises tradicionais
                    all_keypoints.append(frame_keypoints[0])
                    
                    # Armazena keypoints de todas as pessoas para novas análises
                    all_persons_keypoints.append(frame_keypoints)
                    
                    # Armazena confiança da primeira pessoa ou média se houver várias
                    if frame_conf is not None:
                        if len(frame_conf.shape) == 2:  # Se for 2D (várias pessoas)
                            confidence_scores.append(frame_conf[0])
                        else:  # Se for 1D (uma pessoa apenas)
                            confidence_scores.append(frame_conf)
                    else:
                        confidence_scores.append(None)
                    
                    # Conta pessoas detectadas
                    persons_count.append(len(frame_keypoints))
                else:
                    all_keypoints.append(None)
                    all_persons_keypoints.append([])
                    confidence_scores.append(None)
                    persons_count.append(0)
                
            except Exception as e:
                print(f"Erro no frame {idx}: {str(e)}")
                all_keypoints.append(None)
                all_persons_keypoints.append([])
                confidence_scores.append(None)
                persons_count.append(0)
        
        # Suavização temporal
        smoothed_keypoints = None
        if apply_smoothing and any(kp is not None for kp in all_keypoints):
            print("Aplicando suavização temporal...")
            try:
                valid_indices = [i for i, kp in enumerate(all_keypoints) if kp is not None]
                valid_keypoints = [all_keypoints[i] for i in valid_indices]
                
                smoothed_valid = aplicar_suavizacao_temporal(valid_keypoints, window_size)
                
                smoothed_keypoints = [None] * len(all_keypoints)
                for idx, smooth_kp in zip(valid_indices, smoothed_valid):
                    if smooth_kp is not None and len(smooth_kp) > 0:
                        smoothed_keypoints[idx] = smooth_kp
                        np.save(smoothed_dir / f"frame_{idx:05d}.npy", smooth_kp)
            except Exception as e:
                print(f"Erro na suavização temporal: {str(e)}")
                smoothed_keypoints = None
        
        # Cálculo de métricas
        print("Calculando métricas avançadas...")

        # Métricas dos dados brutos (sempre geradas)
        metrics_raw_df = calcular_metricas_avancadas(all_keypoints, None, confidence_scores)
        metrics_raw_df['total_pessoas'] = persons_count

        # Métricas dos dados suavizados (se aplicável)
        metrics_smoothed_df = None
        if apply_smoothing and smoothed_keypoints is not None:
            metrics_smoothed_df = calcular_metricas_avancadas(smoothed_keypoints, None, confidence_scores)
            metrics_smoothed_df['total_pessoas'] = persons_count

        # Salva CSVs separados
        metrics_raw_df.to_csv(output_dir / "raw_metrics.csv", index=False)
        if metrics_smoothed_df is not None:
            metrics_smoothed_df.to_csv(output_dir / "smoothed_metrics.csv", index=False)

        # Gera relatórios independentes
        gerar_relatorio_metricas(metrics_raw_df, output_dir, "raw")
        if metrics_smoothed_df is not None:
            gerar_relatorio_metricas(metrics_smoothed_df, output_dir, "smoothed")
        
    except Exception as e:
        print(f"Erro geral na extração de keypoints: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
def pipeline(video_path, output_dir, frame_rate=5, apply_smoothing=True, window_size=5):
    """Pipeline para processamento de um único vídeo"""
    try:
        # Extração de frames
        print(f"Extraindo frames do vídeo {video_path}...")
        frames = extrair_frames(video_path, frame_rate)
        if not frames:
            return {"status": "error", "message": "Nenhum frame extraído", "metrics_df": None}
        
        # Processamento de keypoints
        print(f"Processando keypoints com {'suavização' if apply_smoothing else 'configuração padrão'}...")
        metrics_df = extract_keypoints_extended(
            frames, output_dir, apply_smoothing=apply_smoothing, window_size=window_size
        )
        
        if metrics_df is None:
            return {"status": "error", "message": "Erro no processamento de keypoints", "metrics_df": None}
        
        # Adiciona informação do vídeo ao DataFrame
        video_name = Path(video_path).stem
        metrics_df['video'] = video_name
        
        # Geração de relatório individual
        nome_video = Path(video_path).stem
        relatorio = gerar_relatorio_metricas(metrics_df, output_dir, nome_video)
        
        return {
            "status": "success",
            "metrics_df": metrics_df,
            "report": relatorio,
            "frames_processed": len(frames),
            "output_location": output_dir
        }
        
    except Exception as e:
        print(f"Erro: {str(e)}")
        return {"status": "error", "message": f"Erro: {str(e)}", "metrics_df": None}


def main():
    RAW_DIR = Path("data/raw")
    PROCESSED_DIR = Path("data/processed")
    FRAME_RATE = 5

    if not RAW_DIR.exists():
        print(f"Erro: Diretório {RAW_DIR} não encontrado.")
        return False

    videos = list(RAW_DIR.glob("*/*.mp4"))
    if not videos:
        print("Nenhum vídeo encontrado em data/raw/.")
        return False

    # Lista para armazenar métricas de todos os vídeos
    all_metrics_dfs = []
    
    for video_path in videos:
        class_name = video_path.parent.name
        output_path = PROCESSED_DIR / "keypoints" / class_name / video_path.stem
        print(f"\nProcessando: {video_path.name}")
        
        result = pipeline(str(video_path), str(output_path), frame_rate=FRAME_RATE)
        
        # Adiciona métricas ao conjunto global se sucesso
        if result["status"] == "success" and result["metrics_df"] is not None:
            # Adiciona informação da classe ao DataFrame
            result["metrics_df"]["classe"] = class_name
            all_metrics_dfs.append(result["metrics_df"])
    
    # Gera relatório geral se há métricas disponíveis
    if all_metrics_dfs:
        # Combina todos os DataFrames
        combined_metrics = pd.concat(all_metrics_dfs, ignore_index=True)
        
        # Salva métricas consolidadas
        metrics_dir = PROCESSED_DIR / "metricas_consolidadas"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        combined_metrics.to_csv(metrics_dir / "all_videos_metrics.csv", index=False)
        
        # Gera relatório consolidado
        gerar_relatorio_metricas(combined_metrics, metrics_dir)
        
        # Gera visualizações consolidadas
        try:
            gerar_visualizacoes_metricas(combined_metrics, metrics_dir / "visualizacoes")
        except Exception as e:
            print(f"Aviso: Não foi possível gerar visualizações consolidadas - {str(e)}")
        
        # Gera análises por classe
        try:
            # Agrupar por classe e calcular médias
            class_metrics = combined_metrics.groupby('classe').mean(numeric_only=True).reset_index()
            class_metrics.to_csv(metrics_dir / "class_average_metrics.csv", index=False)
            
            # Relatório por classe
            for classe in class_metrics['classe'].unique():
                class_data = combined_metrics[combined_metrics['classe'] == classe]
                gerar_relatorio_metricas(class_data, metrics_dir, f"classe_{classe}")
        except Exception as e:
            print(f"Aviso: Erro ao gerar análises por classe - {str(e)}")

    print("\nProcessamento concluído para todos os vídeos.")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)