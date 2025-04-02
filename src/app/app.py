import torch
import asyncio
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())
    
import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import importlib
import importlib.util

# Adicionar o diretório raiz ao PATH para permitir imports relativos
import os
import sys
from pathlib import Path

# Determinar o caminho raiz do projeto
# Assumindo que este arquivo está em src/app/app.py
current_path = Path(os.path.abspath(__file__))
root_dir = current_path.parent.parent.parent  # Subir três níveis: file -> app -> src -> raiz
sys.path.append(str(root_dir))

# Agora podemos importar módulos de outras pastas
from src.preprocessing.dataProcesser import download_youtube_video, extrair_frames, pre_processar_frame
from src.preprocessing.keypoints import extract_keypoints_from_frames

# Configuração da página Streamlit
st.set_page_config(
    page_title="Detector de Assaltos em Vídeos de Segurança",
    page_icon="🎥",
    layout="wide"
)

# Função para visualizar frames com keypoints sobrepostos (simulação)
def visualize_frame_with_keypoints(frame, keypoints=None):
    img = frame.copy()
    if keypoints is not None:
        # Visualização de keypoints - este é um placeholder
        # Será substituído com a visualização real dos keypoints quando integrado com o modelo
        height, width = img.shape[:2]
        for person in keypoints:
            for x, y in person:
                x_px, y_px = int(x * width), int(y * height)
                cv2.circle(img, (x_px, y_px), 5, (0, 255, 0), -1)
            
            # Conectar keypoints para formar um esqueleto (isso é um placeholder)
            # Será substituído com conexões reais dos keypoints do seu modelo
    return img

# Interface Streamlit
st.title("🎥 Detector de Assaltos em Vídeos de Câmeras de Segurança")
st.markdown("""
Esta aplicação permite baixar, processar e detectar assaltos em vídeos de câmeras de segurança.
Insira a URL de um vídeo do YouTube ou faça upload de um vídeo para análise.
""")

# Criar abas para organizar a interface
tab1, tab2, tab3, tab4 = st.tabs(["Carregamento de Vídeo", "Pré-processamento", "Extração de Poses", "Classificação"])

# Aba 1: Carregamento de Vídeo
with tab1:
    st.header("Carregamento de Vídeo")
    
    # Input de URL do YouTube
    youtube_url = st.text_input("URL do vídeo do YouTube", "")
    
    # Upload de vídeo
    uploaded_file = st.file_uploader("Ou faça upload de um vídeo", type=["mp4", "avi", "mov"])
    
    # Botão para baixar/carregar o vídeo
    if st.button("Carregar Vídeo"):
        if youtube_url:
            with st.spinner("Baixando vídeo do YouTube..."):
                try:
                    # Usar a função existente do seu código
                    video_path, video_title = download_youtube_video(youtube_url)
                    st.session_state.video_path = video_path
                    st.session_state.video_title = video_title
                    st.success(f"Vídeo baixado com sucesso: {video_title}")
                    st.video(video_path)
                except Exception as e:
                    st.error(f"Erro ao baixar o vídeo: {str(e)}")
        
        elif uploaded_file:
            with st.spinner("Salvando vídeo enviado..."):
                try:
                    temp_dir = tempfile.mkdtemp()
                    video_path = os.path.join(temp_dir, "uploaded_video.mp4")
                    with open(video_path, "wb") as f:
                        f.write(uploaded_file.read())
                    st.session_state.video_path = video_path
                    st.session_state.video_title = "Vídeo enviado"
                    st.success("Vídeo enviado com sucesso!")
                    st.video(video_path)
                except Exception as e:
                    st.error(f"Erro ao salvar o vídeo: {str(e)}")
    
    # Exibir informações do vídeo se disponível
    if 'video_path' in st.session_state and 'video_title' in st.session_state:
        st.subheader(f"Vídeo Carregado: {st.session_state.video_title}")
        
        # Extrair informações básicas do vídeo
        cap = cv2.VideoCapture(st.session_state.video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        duration = frame_count / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Exibir informações em colunas
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total de Frames", f"{frame_count}")
        with col2:
            st.metric("FPS", f"{fps}")
        with col3:
            st.metric("Duração", f"{duration:.2f} segundos")
        
        st.metric("Resolução", f"{width}x{height}")

# Aba 2: Pré-processamento
with tab2:
    st.header("Pré-processamento de Frames")
    
    if 'video_path' not in st.session_state:
        st.warning("Carregue um vídeo primeiro na aba 'Carregamento de Vídeo'")
    else:
        # Parâmetros de pré-processamento
        st.subheader("Parâmetros de Extração de Frames")
        frame_interval = st.slider("Intervalo de frames para análise", 1, 30, 5, 
                                  help="Extrair 1 frame a cada N frames do vídeo")
        
        # Opções de pré-processamento
        st.subheader("Opções de Pré-processamento")
        col1, col2 = st.columns(2)
        with col1:
            equalizar = st.checkbox("Equalizar histograma", value=True)
            remover_ruido = st.checkbox("Remover ruído", value=True)
        with col2:
            tracar_contorno = st.checkbox("Traçar contorno", value=True)
            tamanho = st.selectbox("Tamanho de saída", [
                "320x240", "640x480", "800x600", "1280x720"
            ], index=1)
            width, height = map(int, tamanho.split("x"))
        
        # Botão para iniciar o pré-processamento
        if st.button("Extrair e Pré-processar Frames"):
            with st.spinner("Extraindo frames..."):
                try:
                    # Usar a função existente do seu código
                    frames = extrair_frames(st.session_state.video_path, f=frame_interval)
                    st.session_state.frames = frames
                    st.success(f"Extração concluída! {len(frames)} frames extraídos.")
                    
                    # Pré-processar frames
                    frames_processados = []
                    for frame in frames:
                        processed = pre_processar_frame(
                            frame,
                            tamanho=(width, height),
                            equalizar=equalizar,
                            remover_ruido=remover_ruido,
                            tracar_contorno=tracar_contorno
                        )
                        frames_processados.append(processed)
                    
                    st.session_state.frames_processados = frames_processados
                    
                    # Mostrar alguns frames de exemplo
                    st.subheader("Visualização de Frames")
                    num_frames_to_show = min(5, len(frames))
                    cols = st.columns(num_frames_to_show)
                    
                    for i, col in enumerate(cols):
                        idx = i * len(frames) // num_frames_to_show
                        with col:
                            # Converter frame em escala de cinza para RGB para exibição
                            if len(frames_processados[idx].shape) == 2:  # Se for escala de cinza
                                display_frame = cv2.cvtColor(frames_processados[idx], cv2.COLOR_GRAY2RGB)
                            else:
                                display_frame = frames_processados[idx]
                            
                            st.image(display_frame, caption=f"Frame {idx}", use_column_width=True)
                    
                except Exception as e:
                    st.error(f"Erro no pré-processamento: {str(e)}")

# Aba 3: Extração de Poses
with tab3:
    st.header("Extração de Keypoints (Poses)")
    
    if 'frames' not in st.session_state:
        st.warning("Extraia os frames primeiro na aba 'Pré-processamento'")
    else:
        st.info("Esta etapa executará o modelo YOLO para detecção de poses e extrairá os keypoints de cada frame.")
        
        # Criar caminhos de diretório relativos ao diretório raiz do projeto
        default_output_dir = os.path.join(str(root_dir), "data", "processed", "keypoints", "temp")
        
        # Diretório para salvar os keypoints
        output_dir = st.text_input(
            "Diretório para salvar os keypoints", 
            value=default_output_dir,
            help="Os keypoints serão salvos neste diretório como arquivos .npy"
        )
        
        # Botão para iniciar a extração de keypoints
        if st.button("Extrair Keypoints"):
            # Criar diretório se não existir
            os.makedirs(output_dir, exist_ok=True)
            
            with st.spinner("Extraindo keypoints dos frames..."):
                try:
                    # Usar a função existente para extrair keypoints
                    extract_keypoints_from_frames(st.session_state.frames, output_dir)
                    st.session_state.keypoints_dir = output_dir
                    st.success(f"Keypoints extraídos com sucesso e salvos em: {output_dir}")
                    
                    # Mostrar alguns frames com keypoints sobrepostos (simulação)
                    st.subheader("Visualização de Keypoints (Simulação)")
                    
                    # Carregar os primeiros arquivos .npy para visualização
                    keypoints_files = sorted(Path(output_dir).glob("*.npy"))
                    if keypoints_files:
                        keypoints_samples = []
                        for i in range(min(3, len(keypoints_files))):
                            sample_keypoints = np.load(keypoints_files[i])
                            keypoints_samples.append(sample_keypoints)
                        
                        # Mostrar frames com keypoints
                        cols = st.columns(len(keypoints_samples))
                        for i, (col, kpts) in enumerate(zip(cols, keypoints_samples)):
                            with col:
                                # Sobrepor keypoints nos frames originais
                                frame_with_keypoints = visualize_frame_with_keypoints(
                                    st.session_state.frames[i], kpts
                                )
                                st.image(frame_with_keypoints, caption=f"Frame {i} com keypoints", use_column_width=True)
                    
                except Exception as e:
                    st.error(f"Erro na extração de keypoints: {str(e)}")

# Aba 4: Classificação
with tab4:
    st.header("Detecção de Assaltos")
    
    if 'keypoints_dir' not in st.session_state:
        st.warning("Extraia os keypoints primeiro na aba 'Extração de Poses'")
    else:
        st.info("Esta etapa utilizará um modelo de classificação para identificar assaltos nos keypoints extraídos.")
        
        # Seleção do modelo
        model_type = st.selectbox(
            "Modelo de Classificação", 
            ["LSTM", "GRU", "BiLSTM", "CNN+LSTM"],
            help="Selecione o tipo de modelo para detecção de assaltos"
        )
        
        # Parâmetros do modelo
        st.subheader("Parâmetros do Modelo")
        seq_length = st.slider("Comprimento da sequência", 5, 50, 20)
        threshold = st.slider("Limiar de confiança", 0.0, 1.0, 0.7, 
                            help="Limiar acima do qual um evento será classificado como assalto")
        
        # Botão para classificar eventos
        if st.button("Detectar Assaltos"):
            with st.spinner("Analisando vídeo para detecção de assaltos..."):
                # Esta seção será integrada com seu modelo real
                st.info("Esta funcionalidade será integrada com seu modelo de classificação.")
                
                # Placeholder para visualização dos resultados (simulação)
                st.subheader("Resultados da Detecção (Simulação)")
                
                # Simular alguns resultados
                resultados = []
                
                for i in range(len(st.session_state.frames) // seq_length):
                    # Simulação de classificação - será substituída pelo seu modelo real
                    confianca = np.random.random()  # Simular probabilidade de assalto
                    
                    resultados.append({
                        "sequencia": i,
                        "frames": f"{i*seq_length} - {(i+1)*seq_length-1}",
                        "tempo_inicio": f"{(i*seq_length/30):.2f}s",
                        "tempo_fim": f"{((i+1)*seq_length-1)/30:.2f}s",
                        "assalto_detectado": confianca > threshold,
                        "confianca": confianca
                    })
                
                # Mostrar tabela de resultados
                df_resultados = pd.DataFrame(resultados)
                
                # Filtrar apenas sequências com assaltos detectados
                df_assaltos = df_resultados[df_resultados["assalto_detectado"] == True]
                
                if len(df_assaltos) > 0:
                    st.success(f"Foram detectados possíveis assaltos em {len(df_assaltos)} segmentos do vídeo!")
                    st.dataframe(df_assaltos[["sequencia", "tempo_inicio", "tempo_fim", "confianca"]])
                else:
                    st.info("Nenhum assalto foi detectado neste vídeo.")
                
                # Mostrar gráfico de confiança ao longo do tempo
                st.subheader("Nível de Confiança ao Longo do Vídeo")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(df_resultados.index, df_resultados["confianca"], marker='o', linestyle='-')
                ax.axhline(y=threshold, color='r', linestyle='--', label=f'Limiar ({threshold})')
                ax.set_xlabel('Segmento de Vídeo')
                ax.set_ylabel('Confiança de Detecção')
                ax.set_title('Probabilidade de Assalto ao Longo do Tempo')
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.7)
                st.pyplot(fig)
                
                # Opção para baixar o relatório
                st.subheader("Relatório")
                csv = df_resultados.to_csv(index=False)
                st.download_button(
                    label="Baixar relatório em CSV",
                    data=csv,
                    file_name=f"relatorio_assaltos_{st.session_state.video_title.replace(' ', '_')}.csv",
                    mime='text/csv',
                )

# Adicionar informações sobre o projeto na barra lateral
st.sidebar.header("Parâmetros do Projeto")

st.sidebar.subheader("Detecção de Poses")
st.sidebar.selectbox("Modelo de Detecção", ["YOLOv8-Pose (nano)", "YOLOv8-Pose (small)", "MediaPipe"])

# Sobre o projeto
st.sidebar.header("Sobre o Projeto")
st.sidebar.markdown("""
**Objetivo:** Detectar e classificar assaltos em vídeos de câmeras de segurança usando:
- Estimativa de poses humanas (YOLOv8)
- Redes Neurais Recorrentes (RNNs)

**Características detectáveis em assaltos:**
- Movimentos bruscos
- Padrões de interação entre pessoas
- Posturas indicativas de ameaça
- Comportamentos não naturais
""")

# Footer
st.markdown("---")
st.markdown("Desenvolvido para detecção automática de assaltos em câmeras de segurança")