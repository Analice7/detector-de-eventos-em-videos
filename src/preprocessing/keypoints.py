from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import sys
from src.preprocessing.extract_frames import extrair_frames #, pre_processar_frame

def extract_keypoints_from_frames(frames: list, output_dir: str) -> None:

    model = YOLO('yolov8n-pose.pt')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, frame in enumerate(frames):  
        # Detecção de poses
        results = model(frame, verbose=False)
        
        if results[0].keypoints is not None:
            keypoints = results[0].keypoints.xyn.cpu().numpy()
            np.save(output_dir / f"frame_{idx:05d}.npy", keypoints)

def pipeline(video_path: str, output_dir: str, frame_rate: int = 5):

    if not Path(video_path).exists():
        print(f"Erro: Vídeo {video_path} não encontrado!")
        sys.exit(1)

    # 1. Extrai frames
    try:
        frames = extrair_frames(video_path, f=frame_rate)
    except Exception as e:
        print(f"Erro na extração: {str(e)}")
        sys.exit(1)
    
    # 2. Pré-processamento
    # frames_processados = [
    #     pre_processar_frame(
    #         frame,
    #         tamanho=(640, 480),
    #         equalizar=True,
    #         remover_ruido=True,
    #         tracar_contorno=True
    #     ) for frame in frames
    # ]
    
    # 3. Extrai keypoints
    extract_keypoints_from_frames(frames, output_dir)
    print(f"Keypoints salvos em: {output_dir}")

if __name__ == "__main__":

    RAW_DIR = Path("data/raw")
    PROCESSED_DIR = Path("data/processed")
    FRAME_RATE = 5

    # Para vídeo local
    for video_path in RAW_DIR.glob("*/*.mp4"):
        class_name = video_path.parent.name 
        output_path = PROCESSED_DIR / "keypoints" / class_name / video_path.stem
        
        print(f"\nProcessando: {video_path.name}")
        pipeline(
            video_path=str(video_path),
            output_dir=str(output_path),
            frame_rate=FRAME_RATE
        )
    
    # Para vídeo do YouTube
    """
    video_path, _ = download_youtube_video("URL_DO_YOUTUBE")
    pipeline(video_path, "data/keypoints/youtube_video")
    """