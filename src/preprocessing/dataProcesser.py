import cv2
from pytube import YouTube
import tempfile

def download_youtube_video(url):
    yt = YouTube(url)   
    video_resolucao = yt.streams.filter(progressive=True, file_extension="mp4").order_by('resolution').desc().first()
    
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_file.close()
    
    # Download the video
    video_resolucao.download(filename=temp_file.name)
    
    return temp_file.name, yt.title

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