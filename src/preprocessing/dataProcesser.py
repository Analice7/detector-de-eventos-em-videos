import cv2
from pytube import YouTube
import tempfile
import os
import logging
from pathlib import Path
import uuid

def download_youtube_video(url):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Caminho absoluto para o cookies.txt
    cookies_path = Path(__file__).parent.parent.parent / "cookies.txt"
    cookies_str = str(cookies_path.resolve())

    try:
        import importlib.util
        if importlib.util.find_spec("yt_dlp"):
            logger.info("Tentando baixar com yt-dlp com cookies...")
            from yt_dlp import YoutubeDL

            # Cria um nome de arquivo único no diretório temporário
            temp_filename = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.mp4")

            ydl_opts = {
                'format': 'best[ext=mp4]/best',
                'outtmpl': temp_filename,
                'quiet': True,
                'cookiefile': cookies_str,
            }

            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                title = info.get('title', 'Video')

            if os.path.exists(temp_filename) and os.path.getsize(temp_filename) > 100 * 1024:
                logger.info(f"Vídeo baixado com sucesso via yt-dlp: {title}")
                return temp_filename, title
            else:
                logger.error("Arquivo de vídeo corrompido ou incompleto")

    except Exception as e:
        logger.error(f"Erro ao baixar com yt-dlp: {str(e)}")
    
    return None, None

def extrair_frames(filepath, f=5):
    if filepath:
        cap = cv2.VideoCapture(filepath)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        success = True
        frame_cont = 0
        while success:
            success, frame = cap.read()
            if not success:
                break
            if frame_cont % f == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            frame_cont += 1
        cap.release()
        return frames

def pre_processar_frame(frame, tamanho=(640, 480), equalizar=True, remover_ruido=True, tracar_contorno = True):
    if frame is None:
        return None

    # Redimensionar para tamanho padrão
    frame_processado = cv2.resize(frame, tamanho)
    
    # Converter para escala de cinza
    frame_cinza = cv2.cvtColor(frame_processado, cv2.COLOR_BGR2GRAY)

    # Aplicar contorno usando Canny Edge
    if tracar_contorno:
        frame_cinza = cv2.Canny(frame_cinza, 100, 200)
    
    # Equalização de histograma para melhorar contraste
    if equalizar:
        frame_cinza = cv2.equalizeHist(frame_cinza)
    
    # Redução de ruído
    if remover_ruido:
        frame_cinza = cv2.GaussianBlur(frame_cinza, (5, 5), 0)
    
    return frame_cinza