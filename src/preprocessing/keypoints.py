import numpy as np
import cv2
from pathlib import Path
import pandas as pd
from ultralytics import YOLO
from ..metrics.keypoints import (
    gerar_visualizacoes_metricas,
    gerar_relatorio_metricas,
    calcular_metricas_avancadas,
)
import sys
from src.preprocessing.dataProcesser import extrair_frames, pre_processar_frame

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

def extract_keypoints_extended(frames, video_name, output_base_dir,  class_name,apply_smoothing=False, window_size=5):
    """Processa os frames e extrai keypoints"""
    try:
        model = YOLO('yolov8n-pose.pt')
        
        # Novas estruturas de diretórios
        base_dir = Path(output_base_dir)
        
        # Diretório para dados não suavizados
        no_smoothed_dir = base_dir / "no_smoothed" / class_name / video_name
        no_smoothed_dir.mkdir(parents=True, exist_ok=True)
        
        # Diretório para dados suavizados (se aplicável)
        smoothed_dir = None
        if apply_smoothing:
            smoothed_dir = base_dir / "smoothed" / class_name / video_name
            smoothed_dir.mkdir(parents=True, exist_ok=True)
        
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
                
                # Salva e armazena keypoints
                if len(frame_keypoints) > 0:
                    # Salva os keypoints de todas as pessoas detectadas
                    np.save(no_smoothed_dir / f"frame_{idx:05d}.npy", frame_keypoints)
                    
                    # Armazena keypoints da primeira pessoa para análises tradicionais
                    all_keypoints.append(frame_keypoints[0])
                    
                    # Armazena keypoints de todas as pessoas para novas análises
                    all_persons_keypoints.append(frame_keypoints)
                    
                    # Armazena confiança da primeira pessoa
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
        if apply_smoothing and any(len(frame_kps) > 0 for frame_kps in all_persons_keypoints):
            print("Aplicando suavização temporal a TODAS as pessoas...")
            try:
                # 1. Determina o número máximo de pessoas detectadas em qualquer frame
                max_pessoas = max(len(frame_kps) for frame_kps in all_persons_keypoints)
                
                # 2. Suaviza cada pessoa individualmente
                pessoas_suavizadas = []
                for pessoa_idx in range(max_pessoas):
                    # Coleta a sequência dessa pessoa em todos os frames
                    sequencia_pessoa = []
                    for frame_kps in all_persons_keypoints:
                        if pessoa_idx < len(frame_kps):
                            sequencia_pessoa.append(frame_kps[pessoa_idx])
                        else:
                            sequencia_pessoa.append(np.array([]))  # Placeholder se não detectado
                    
                    # Aplica suavização a essa pessoa
                    pessoa_suavizada = aplicar_suavizacao_temporal(sequencia_pessoa, window_size)
                    pessoas_suavizadas.append(pessoa_suavizada)
                
                # 3. Reorganiza os dados por frame
                smoothed_by_frame = []
                for frame_idx in range(len(all_persons_keypoints)):
                    frame_data = []
                    for pessoa_idx in range(max_pessoas):
                        if frame_idx < len(pessoas_suavizadas[pessoa_idx]):
                            kp = pessoas_suavizadas[pessoa_idx][frame_idx]
                            if kp.size > 0:  # Filtra arrays vazios
                                frame_data.append(kp)
                    
                    smoothed_by_frame.append(np.array(frame_data) if frame_data else np.array([]))
                
                # 4. Salva os resultados
                for idx, frame_kps in enumerate(smoothed_by_frame):
                    if len(frame_kps) > 0:
                        np.save(smoothed_dir / f"frame_{idx:05d}.npy", frame_kps)
                
                # Mantém compatibilidade com o resto do código (opcional)
                smoothed_keypoints = [frame_kps[0] if len(frame_kps) > 0 else None for frame_kps in smoothed_by_frame]
                
            except Exception as e:
                print(f"Erro na suavização multi-pessoa: {str(e)}")
                smoothed_keypoints = None
        
        # Retorna dados para geração de relatórios
        return {
            "raw_keypoints": all_keypoints,
            "smoothed_keypoints": smoothed_keypoints,
            "confidence_scores": confidence_scores,
            "persons_count": persons_count
        }
        
    except Exception as e:
        print(f"Erro geral na extração de keypoints: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
def pipeline(video_path, output_base_dir, class_name, frame_rate=15, apply_smoothing=True, window_size=5):
    """Pipeline para processamento de um único vídeo"""
    try:
        # Extração de frames
        print(f"Extraindo frames do vídeo {video_path}...")
        frames = extrair_frames(video_path, frame_rate)
        if not frames:
            return {"status": "error", "message": "Nenhum frame extraído"}
        
        # Pré-processamento
        # frames = [
        #     pre_processar_frame(
        #         frame,
        #         tamanho=(640, 480),
        #         equalizar=True,
        #         remover_ruido=True,
        #         tracar_contorno=True
        #     ) for frame in frames
        # ]
        
        # Nome do vídeo para uso na estrutura de diretórios
        video_name = Path(video_path).stem
        
        # Processamento de keypoints
        print(f"Processando keypoints com {'suavização' if apply_smoothing else 'configuração padrão'}...")
        result = extract_keypoints_extended(frames, video_name, output_base_dir, class_name, apply_smoothing=apply_smoothing, window_size=window_size)
        
        if result is None:
            return {"status": "error", "message": "Erro no processamento de keypoints"}
        
        # Criação de DataFrames para métricas
        raw_metrics_df = calcular_metricas_avancadas(
            result["raw_keypoints"], None, result["confidence_scores"]
        )
        raw_metrics_df['total_pessoas'] = result["persons_count"]
        raw_metrics_df['video'] = video_name
        
        # Métricas dos dados suavizados (se aplicável)
        smoothed_metrics_df = None
        if apply_smoothing and result["smoothed_keypoints"] is not None:
            smoothed_metrics_df = calcular_metricas_avancadas(
                result["smoothed_keypoints"], None, result["confidence_scores"]
            )
            smoothed_metrics_df['total_pessoas'] = result["persons_count"]
            smoothed_metrics_df['video'] = video_name
        
        return {
            "status": "success",
            "raw_metrics_df": raw_metrics_df,
            "smoothed_metrics_df": smoothed_metrics_df,
            "frames_processed": len(frames)
        }
        
    except Exception as e:
        print(f"Erro: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": f"Erro: {str(e)}"}


def main():
    RAW_DIR = Path("data/raw")
    PROCESSED_DIR = Path("data/processed")
    KEYPOINTS_DIR = PROCESSED_DIR / "keypoints"
    FRAME_RATE = 15
    
    # Aplicar suavização (será feito para todos os vídeos)
    APPLY_SMOOTHING = True
    WINDOW_SIZE = 5
    
    if not RAW_DIR.exists():
        print(f"Erro: Diretório {RAW_DIR} não encontrado.")
        return False

    videos = list(RAW_DIR.glob("*/*.mp4"))
    if not videos:
        print("Nenhum vídeo encontrado em data/raw/.")
        return False

    # Listas para armazenar métricas de todos os vídeos
    all_raw_metrics = []
    all_smoothed_metrics = []
    
    for video_path in videos:
        class_name = video_path.parent.name  # "assault" ou "normal"
        print(f"\nProcessando: {video_path.name} (Classe: {class_name})")
        
        result = pipeline(
            str(video_path), 
            str(KEYPOINTS_DIR),
            class_name, 
            frame_rate=FRAME_RATE, 
            apply_smoothing=APPLY_SMOOTHING,
            window_size=WINDOW_SIZE
        )
        
        # Adiciona métricas ao conjunto global se sucesso
        if result["status"] == "success":
            # Adiciona informação da classe ao DataFrame
            result["raw_metrics_df"]["classe"] = class_name
            all_raw_metrics.append(result["raw_metrics_df"])
            
            if result["smoothed_metrics_df"] is not None:
                result["smoothed_metrics_df"]["classe"] = class_name
                all_smoothed_metrics.append(result["smoothed_metrics_df"])
    
    # Diretório para relatórios gerais
    reports_dir = PROCESSED_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Gera relatório geral para dados não suavizados
    if all_raw_metrics:
        # Combina todos os DataFrames
        combined_raw_metrics = pd.concat(all_raw_metrics, ignore_index=True)
        
        # Salva métricas consolidadas
        combined_raw_metrics.to_csv(reports_dir / "no_smoothed_metrics.csv", index=False)
        
        # Gera relatório consolidado
        gerar_relatorio_metricas(combined_raw_metrics, reports_dir, "no_smoothed")
        
        # Gera visualizações consolidadas
        try:
            gerar_visualizacoes_metricas(combined_raw_metrics, reports_dir / "no_smoothed_visualizations")
        except Exception as e:
            print(f"Aviso: Não foi possível gerar visualizações para dados não suavizados - {str(e)}")
    
    # Gera relatório geral para dados suavizados
    if all_smoothed_metrics:
        # Combina todos os DataFrames
        combined_smoothed_metrics = pd.concat(all_smoothed_metrics, ignore_index=True)
        
        # Salva métricas consolidadas
        combined_smoothed_metrics.to_csv(reports_dir / "smoothed_metrics.csv", index=False)
        
        # Gera relatório consolidado
        gerar_relatorio_metricas(combined_smoothed_metrics, reports_dir, "smoothed")
        
        # Gera visualizações consolidadas
        try:
            gerar_visualizacoes_metricas(combined_smoothed_metrics, reports_dir / "smoothed_visualizations")
        except Exception as e:
            print(f"Aviso: Não foi possível gerar visualizações para dados suavizados - {str(e)}")

    print("\nProcessamento concluído para todos os vídeos.")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)