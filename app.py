import torch
import streamlit as st
import time
import pandas as pd
import numpy as np
import altair as alt
import cv2
import tempfile
import os
from pathlib import Path
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

# Importando os módulos necessários do seu código existente
from src.preprocessing.dataProcesser import download_youtube_video, extrair_frames
from src.preprocessing.keypoints import extract_keypoints_from_frames
from src.preprocessing.features import process_video_keypoints

# Função para carregar o modelo treinado
def load_model(model_path='best_lstm_model.pth'):
    try:
        # Importar a classe PoseRNN do módulo train.py
        from src.training.train import PoseRNN, Config
        
        # Determinar o tamanho de entrada a partir dos dados de treinamento
        input_size = 72  # Substitua pelo tamanho correto baseado nas suas características
        
        # Criar modelo com a mesma arquitetura usada no treinamento
        model = PoseRNN(input_size, rnn_type='lstm')
        
        # Carregar os pesos
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        
        # Colocar o modelo em modo de avaliação
        model.eval()
        
        return model
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None

# Função para fazer predições
def predict(model, features, seq_length=10):
    try:
        # Preparar os dados para o modelo
        n_frames, n_features = features.shape
        
        # Criar sequências
        predictions = []
        confidences = []
        
        with torch.no_grad():
            for i in range(0, n_frames - seq_length + 1, seq_length // 2):  # 50% de sobreposição
                seq = features[i:i+seq_length]
                if len(seq) == seq_length:
                    # Converter para tensor
                    seq_tensor = torch.FloatTensor(seq).unsqueeze(0)  # Add batch dimension
                    
                    # Fazer predição
                    output = model(seq_tensor)
                    
                    # Aplicar softmax para obter probabilidades
                    probs = torch.nn.functional.softmax(output, dim=1)
                    
                    # Obter classe com maior probabilidade
                    _, predicted = torch.max(output.data, 1)
                    
                    # Armazenar resultados
                    predictions.append(predicted.item())
                    confidences.append(probs[0, 1].item())  # Probabilidade da classe 1 (agressão)
        
        return predictions, confidences
    except Exception as e:
        st.error(f"Erro na predição: {e}")
        return [], []

def process_video(youtube_url, progress_callback=None):
    try:
        # 1. Download do vídeo
        progress_callback("Baixando vídeo...", 0.1)
        video_path, title = download_youtube_video(youtube_url)
        
        if not video_path:
            raise Exception("Falha ao baixar o vídeo. Verifique se o URL é válido e tente novamente.")
        
        # 2. Extração de frames
        progress_callback("Extraindo frames...", 0.3)
        frames = extrair_frames(video_path, f=5)
        if not frames or len(frames) == 0:
            raise Exception("Nenhum frame extraído do vídeo")
        
        # 3. Extração de keypoints
        progress_callback("Detectando poses humanas...", 0.5)
        temp_dir = tempfile.mkdtemp()
        keypoints_dir = os.path.join(temp_dir, "keypoints")
        os.makedirs(keypoints_dir, exist_ok=True)
        extract_keypoints_from_frames(frames, keypoints_dir)
        
        # Verificar se os keypoints foram extraídos
        keypoints_files = [f for f in os.listdir(keypoints_dir) if f.startswith("frame_")]
        if not keypoints_files:
            raise Exception("Nenhum keypoint extraído dos frames")
        
        # 4. Extração de características
        progress_callback("Extraindo características...", 0.7)
        features_dir = os.path.join(temp_dir, "features")
        os.makedirs(features_dir, exist_ok=True)
        features_dict = process_video_keypoints(keypoints_dir, features_dir)
        
        if not features_dict:
            raise Exception("Falha ao extrair características")
        
        # Preparar sequência de características para classificação
        features_file = os.path.join(features_dir, "features.npy")
        if os.path.exists(features_file):
            features_sequence = np.load(features_file)
        else:
            # Fallback caso o arquivo não exista
            feature_keys = sorted(features_dict.keys())
            if not feature_keys:
                raise Exception("Dicionário de características vazio")
                
            n_frames = len(features_dict[feature_keys[0]])
            features_sequence = np.zeros((n_frames, len(feature_keys)))
            for i, key in enumerate(feature_keys):
                features_sequence[:, i] = features_dict[key]
        
        # 5. Classificação com o modelo
        progress_callback("Classificando comportamento...", 0.9)
        model = load_model()
        if not model:
            raise Exception("Falha ao carregar o modelo")
            
        predictions, confidences = predict(model, features_sequence)
        if not predictions or len(predictions) == 0:
            raise Exception("Nenhuma predição gerada")
        
        # 6. Resultados finais
        progress_callback("Finalizando análise...", 1.0)
        
        # Determinar resultado geral
        avg_confidence = np.mean(confidences) if confidences else 0.5
        
        # Selecionar frame crítico
        if confidences and frames:
            max_confidence_index = min(np.argmax(confidences), len(frames)-1)
            critical_frame = frames[max_confidence_index]
        else:
            critical_frame = frames[-1] if frames else None
        
        return {
            'video_title': title,
            'confidences': confidences,
            'avg_confidence': avg_confidence,
            'frames': frames,
            'critical_frame': critical_frame,
            'max_confidence_index': np.argmax(confidences) if confidences else 0,
            'success': True
        }
    
    except Exception as e:
        st.error(f"Erro no processamento: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

def main():
    # Configuração da página
    st.set_page_config(
        page_title="Detector de Agressões em Vídeos",
        page_icon="🎬",
        layout="wide"
    )
    
    # Título e descrição
    st.title("🎬 Sistema de Detecção de Agressões em Vídeos")
    st.markdown("""
    Este sistema analisa vídeos para detectar possíveis ocorrências de agressões físicas
    utilizando modelos de estimativa de poses humanas e Redes Neurais Recorrentes.
    """)
    
    # Sidebar
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
        4. Classificação por Redes Neurais Recorrentes
        """)
        
        st.divider()
        st.markdown("Desenvolvido para disciplina de Tópicos Especiais em IA")
    
    # Formulário principal
    st.header("Análise de Vídeo")
    
    with st.form(key="video_form"):
        youtube_url = st.text_input(
            "Link do vídeo do YouTube:",
            placeholder="Ex: https://www.youtube.com/watch?v=..."
        )
        
        submit_button = st.form_submit_button(label="Analisar Vídeo")
    
    # Lógica após o envio do formulário
    if submit_button and youtube_url:
        # Exibição do link fornecido
        st.success(f"Link do vídeo recebido: {youtube_url}")
        
        # Processamento completo com uma única barra de progresso
        with st.spinner("Analisando vídeo..."):
            # Container para mostrar o progresso geral
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Callback para atualizar o progresso
            def update_progress(message, progress):
                status_text.text(message)
                progress_bar.progress(progress)
            
            # Processar o vídeo (substituir pela implementação real)
            # Para teste da interface, podemos usar um processamento simulado
            # Processar o vídeo
            try:
                results = process_video(youtube_url, update_progress)
                
                if not results or not results.get('success', False):
                    st.error(f"Erro ao processar o vídeo: {results.get('error', 'Erro desconhecido')}")
                    
                    # Oferecer opção de visualizar dados simulados
                    if st.button("Visualizar exemplo com dados simulados"):
                        # Simular resultados para demonstração
                        status_text.text("Gerando visualização com dados simulados...")
                        time.sleep(1)
                        results = {
                            'video_title': "Simulação - " + youtube_url,
                            'confidences': np.clip(0.5 + 0.5 * np.sin(np.arange(0, 30, 1)/3) + np.random.normal(0, 0.1, 30), 0, 1),
                            'avg_confidence': 0.67,
                            'frames': [None] * 30,  # Placeholder
                            'critical_frame': None,
                            'max_confidence_index': 15,
                            'success': True
                        }
                    else:
                        # Encerrar o processamento aqui se não quiser dados simulados
                        progress_bar.empty()
                        status_text.empty()
                        st.stop()
                
                status_text.text("Análise concluída!")
                
            except Exception as e:
                st.error(f"Erro durante o processamento: {str(e)}")
                progress_bar.empty()
                status_text.empty()
                st.stop()
            
            if not results:
                # Simular resultados para teste da interface
                status_text.text("Simulando resultados para teste...")
                time.sleep(1)
                results = {
                    'video_title': "Vídeo de exemplo",
                    'confidences': np.clip(0.5 + 0.5 * np.sin(np.arange(0, 30, 1)/3) + np.random.normal(0, 0.1, 30), 0, 1),
                    'avg_confidence': 0.67,
                    'frames': [None] * 30,  # Placeholder
                    'critical_frame': None,
                    'max_confidence_index': 15
                }
            
            status_text.text("Análise concluída!")
        
        # Exibição dos resultados
        st.header("Resultados da Análise")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Probabilidade de agressão
            result_probability = results['avg_confidence']
            st.metric(
                label="Probabilidade de Agressão", 
                value=f"{result_probability:.2%}"
            )
            
            # Conclusão baseada na probabilidade
            confidence_threshold = 0.7  # Limiar para classificação
            if result_probability > confidence_threshold:
                st.error("⚠️ **ALERTA:** Comportamento de agressão física detectado!")
            else:
                st.success("✅ Nenhum comportamento de agressão detectado.")
        
        with col2:
            # Gráfico de confiança ao longo do tempo
            st.subheader("Confiança ao longo do vídeo")
            
            chart_data = pd.DataFrame({
                'Frame': np.arange(0, len(results['confidences'])),
                'Confiança': results['confidences']
            })
            
            chart = alt.Chart(chart_data).mark_line().encode(
                x='Frame',
                y=alt.Y('Confiança', scale=alt.Scale(domain=[0, 1]))
            ).properties(height=250)
            
            st.altair_chart(chart, use_container_width=True)
        
        # Seção para frames principais detectados
        st.subheader("Momento crítico detectado")
        
        # Identificando o momento com maior probabilidade
        max_confidence_index = results['max_confidence_index']
        max_confidence_value = results['confidences'][max_confidence_index]
        
        # Se temos um frame crítico real, mostrá-lo
        if results['critical_frame'] is not None:
            # Converter o frame para formato que o streamlit pode exibir
            critical_frame = cv2.cvtColor(results['critical_frame'], cv2.COLOR_BGR2RGB)
            st.image(
                critical_frame,
                caption=f"Momento de maior probabilidade de agressão ({max_confidence_value:.2%}) no frame {max_confidence_index}"
            )
        else:
            # Placeholder para testes da interface
            st.image(
                "https://via.placeholder.com/800x450?text=Momento+Crítico+de+Agressão",
                caption=f"Momento de maior probabilidade ({max_confidence_value:.2%}) no frame {max_confidence_index}"
            )

if __name__ == "__main__":
    main()