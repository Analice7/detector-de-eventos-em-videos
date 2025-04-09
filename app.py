import torch
import streamlit as st
import time
import pandas as pd
import numpy as np
import altair as alt
import cv2
import tempfile
import os
import uuid
from pathlib import Path
import imageio.v2 as imageio  # Novo para salvar GIF

# Configurar o path do torch corretamente
if hasattr(torch, 'classes') and hasattr(torch.classes, '__path__'):
    torch.classes.__path__ = [os.path.join(torch.__path__[0], 'classes')] 

# Importando os módulos necessários do seu código existente
from src.preprocessing.dataProcesser import download_youtube_video, extrair_frames
from src.preprocessing.keypoints import extract_keypoints_extended
from src.preprocessing.features import process_video_keypoints

# Constantes
MODEL_TYPE = 'lstm'  # Modelo da RNN

def salvar_gif(frames, path, fps=5, loop=0):
    if not frames:
        return None
    try:
        duration = 1 / fps
        imageio.mimsave(path, frames, format='GIF', duration=duration, loop=loop)
        return path
    except Exception as e:
        print(f"Erro ao salvar GIF: {e}")
        return None

# Função para carregar o modelo treinado
def load_model(model_path=f'best_{MODEL_TYPE}_model.pth'):
    try:
        from src.training.train import PoseRNN, Config
        Config.HIDDEN_SIZE = 512
        input_size = 72
        model = PoseRNN(input_size, rnn_type=MODEL_TYPE)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

# Função para fazer predições
def predict(model, features, seq_length=64):
    try:
        n_frames, n_features = features.shape
        predictions = []
        confidences = []

        with torch.no_grad():
            for i in range(0, n_frames - seq_length + 1, seq_length // 2):
                seq = features[i:i+seq_length]
                if len(seq) == seq_length:
                    seq_tensor = torch.FloatTensor(seq).unsqueeze(0)
                    output = model(seq_tensor)
                    probs = torch.nn.functional.softmax(output, dim=1)
                    _, predicted = torch.max(output.data, 1)
                    predictions.append(predicted.item())
                    confidences.append(probs[0, 1].item())

        return predictions, confidences
    except Exception as e:
        st.error(f"Erro na predição: {e}")
        import traceback
        st.error(traceback.format_exc())
        return [], []

def process_video(youtube_url, progress_callback=None):
    try:
        progress_callback("Baixando vídeo...", 0.1)
        video_path, title = download_youtube_video(youtube_url)

        if not video_path:
            raise Exception("Falha ao baixar o vídeo. Verifique se o URL é válido e tente novamente.")

        progress_callback("Extraindo frames...", 0.3)
        frames = extrair_frames(video_path, f=5)
        if not frames or len(frames) == 0:
            raise Exception("Nenhum frame extraído do vídeo")

        progress_callback("Detectando poses humanas...", 0.5)
        temp_dir = tempfile.mkdtemp()
        video_name = f"video_{uuid.uuid4().hex}"
        class_name = "unknown"

        keypoints_result = extract_keypoints_extended(
            frames=frames, 
            video_name=video_name, 
            output_base_dir=temp_dir,
            class_name=class_name,
            apply_smoothing=True,
            window_size=5
        )

        if not keypoints_result or not isinstance(keypoints_result, dict):
            raise Exception("Falha na extração de keypoints")

        progress_callback("Extraindo características...", 0.7)
        keypoints_dir = os.path.join(temp_dir, "no_smoothed", class_name, video_name)
        features_dir = os.path.join(temp_dir, "features", class_name, video_name)
        os.makedirs(features_dir, exist_ok=True)

        features_dict = process_video_keypoints(keypoints_dir, features_dir)

        if not features_dict:
            raise Exception("Falha ao extrair características")

        features_file = os.path.join(features_dir, "features.npy")
        if os.path.exists(features_file):
            features_sequence = np.load(features_file)
        else:
            feature_keys = sorted(features_dict.keys())
            if not feature_keys:
                raise Exception("Dicionário de características vazio")
            n_frames = len(features_dict[feature_keys[0]])
            features_sequence = np.zeros((n_frames, len(feature_keys)))
            for i, key in enumerate(feature_keys):
                features_sequence[:, i] = features_dict[key]

        progress_callback("Classificando comportamento...", 0.9)
        model = load_model()
        if not model:
            raise Exception("Falha ao carregar o modelo")

        predictions, confidences = predict(model, features_sequence)
        if not predictions or len(predictions) == 0:
            raise Exception("Nenhuma predição gerada")

        progress_callback("Finalizando análise...", 1.0)
        avg_confidence = np.mean(confidences) if confidences else 0.5

        if confidences and frames:
            max_confidence_index = np.argmax(confidences)
            seq_length = 64
            step = seq_length // 2
            start_idx = max_confidence_index * step
            end_idx = start_idx + seq_length
            window_frames = frames[start_idx:end_idx][:min(seq_length, len(frames) - start_idx)]

            video_clip_path = os.path.join(temp_dir, "critical_clip.gif")
            salvar_gif(window_frames, video_clip_path)
        else:
            video_clip_path = None
            max_confidence_index = 0

        return {
            'video_title': title if title else youtube_url,
            'confidences': confidences,
            'avg_confidence': avg_confidence,
            'frames': frames,
            'critical_clip_path': video_clip_path,
            'max_confidence_index': max_confidence_index,
            'success': True
        }

    except Exception as e:
        st.error(f"Erro no processamento: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return {
            'success': False,
            'error': str(e)
        }

def main():
    st.set_page_config(
        page_title="Detector de Agressões em Vídeos",
        page_icon="🎬",
        layout="wide"
    )

    st.title("🎬 Sistema de Detecção de Agressões em Vídeos")
    st.markdown("""
    Este sistema analisa vídeos para detectar possíveis ocorrências de agressões físicas
    utilizando modelos de estimativa de poses humanas e Redes Neurais Recorrentes.
    """)

    with st.sidebar:
        st.header("Sobre o Sistema")
        st.info("""
        **Projeto de Visão Computacional**

        Este sistema utiliza técnicas avançadas de visão computacional 
        e aprendizado profundo para detectar comportamentos característicos 
        de agressões físicas em vídeos de câmeras de segurança.

        O pipeline de processamento inclui:
        1. Extração de frames do vídeo
        2. Detecção de poses com YOLOv8
        3. Extração de características dos movimentos
        4. Classificação por Rede Neural Recorrente (RNN)
        """)

        st.divider()
        st.markdown("Desenvolvido para disciplina de Tópicos Especiais em IA")

    st.header("Análise de Vídeo")

    with st.form(key="video_form"):
        youtube_url = st.text_input(
            "Link do vídeo do YouTube:",
            placeholder="Ex: https://www.youtube.com/watch?v=..."
        )

        submit_button = st.form_submit_button(label="Analisar Vídeo")

    if submit_button and youtube_url:
        st.success(f"Link do vídeo recebido: {youtube_url}")

        with st.spinner("Analisando vídeo..."):
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(message, progress):
                status_text.text(message)
                progress_bar.progress(progress)

            try:
                results = process_video(youtube_url, update_progress)

                if not results or not results.get('success', False):
                    st.error(f"Erro ao processar o vídeo: {results.get('error', 'Erro desconhecido')}")

                    if st.button("Visualizar exemplo com dados simulados"):
                        status_text.text("Gerando visualização com dados simulados...")
                        time.sleep(1)
                        results = {
                            'video_title': "Simulação - " + youtube_url,
                            'confidences': np.clip(0.5 + 0.5 * np.sin(np.arange(0, 30, 1)/3) + np.random.normal(0, 0.1, 30), 0, 1),
                            'avg_confidence': 0.67,
                            'frames': [None] * 30,
                            'critical_clip_path': None,
                            'max_confidence_index': 15,
                            'success': True
                        }
                    else:
                        progress_bar.empty()
                        status_text.empty()
                        st.stop()

                status_text.text("Análise concluída!")

            except Exception as e:
                st.error(f"Erro durante o processamento: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
                progress_bar.empty()
                status_text.empty()
                st.stop()

            status_text.text("Análise concluída!")

        st.header("Resultados da Análise")

        col1, col2 = st.columns(2)

        with col1:
            result_probability = results['avg_confidence']
            st.metric(
                label="Probabilidade de Agressão", 
                value=f"{result_probability:.2%}"
            )

            confidence_threshold = 0.65
            if result_probability > confidence_threshold:
                st.error("⚠️ **ALERTA:** Comportamento de agressão física detectado!")
            else:
                st.success("✅ Nenhum comportamento de agressão detectado.")

        with col2:
            st.subheader("Momento crítico detectado")
            max_confidence_index = results['max_confidence_index']
            max_confidence = results['confidences'][np.argmax(results['confidences'])]

            if results['critical_clip_path'] is not None:
                st.image(results['critical_clip_path'], caption=f"Momento de maior probabilidade ({max_confidence:.2%}) no frame {max_confidence_index}")
            else:
                st.image(
                    "https://via.placeholder.com/800x450?text=Momento+Crítico+de+Agressão",
                    caption=f"Momento de maior probabilidade ({max_confidence:.2%}) detectado na janela {max_confidence_index}"
                )

if __name__ == "__main__":
    main()