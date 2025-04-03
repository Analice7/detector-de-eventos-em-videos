import streamlit as st
import time
import pandas as pd
import numpy as np
import altair as alt

def main():
    # Configuração da página
    st.set_page_config(
        page_title="Detector de Assaltos em Vídeos",
        page_icon="🎥",
        layout="wide"
    )
    
    # Título e descrição
    st.title("🎥 Sistema de Detecção de Assaltos em Vídeos")
    st.markdown("""
    Este sistema analisa vídeos para detectar possíveis ocorrências de assaltos
    utilizando modelos de estimativa de poses humanas e Redes Neurais Recorrentes.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("Sobre o Sistema")
        st.info("""
        **Projeto de Visão Computacional**
        
        Este sistema utiliza técnicas avançadas de visão computacional 
        e aprendizado profundo para detectar comportamentos característicos 
        de assaltos em vídeos de câmeras de segurança.
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
            
            # Simulação do progresso (substitua por seu processamento real)
            for i in range(100):
                # Atualiza barra de progresso
                progress_bar.progress(i + 1)
                
                # Atualiza texto de status de forma mais simples
                if i < 25:
                    status_text.text("Baixando vídeo...")
                elif i < 75:
                    status_text.text("Processando vídeo...")
                else:
                    status_text.text("Finalizando análise...")
                
                time.sleep(0.05)
            
            status_text.text("Análise concluída!")
        
        # Exibição dos resultados
        st.header("Resultados da Análise")
        
        # Simulação dos resultados - substitua por resultados reais quando integrar
        col1, col2 = st.columns(2)
        
        with col1:
            # Probabilidade de assalto - substitua por resultado real
            result_probability = 0.83  # Valor de exemplo
            st.metric(
                label="Probabilidade de Assalto", 
                value=f"{result_probability:.2%}"
            )
            
            # Conclusão baseada na probabilidade (usando um limite fixo pré-definido)
            confidence_threshold = 0.7  # Definido pelos desenvolvedores, não pelo usuário
            if result_probability > confidence_threshold:
                st.error("⚠️ **ALERTA:** Comportamento característico de assalto detectado!")
            else:
                st.success("✅ Nenhum comportamento suspeito detectado.")
        
        with col2:
            # Gráfico de confiança ao longo do tempo (simulado)
            st.subheader("Confiança ao longo do vídeo")
            
            # Dados simulados - substitua pelos dados reais quando integrar
            chart_data = pd.DataFrame({
                'Tempo (s)': np.arange(0, 30, 1),
                'Confiança': np.clip(0.5 + 0.5 * np.sin(np.arange(0, 30, 1)/3) + np.random.normal(0, 0.1, 30), 0, 1)
            })
            
            chart = alt.Chart(chart_data).mark_line().encode(
                x='Tempo (s)',
                y=alt.Y('Confiança', scale=alt.Scale(domain=[0, 1]))
            ).properties(height=250)
            
            st.altair_chart(chart, use_container_width=True)
        
        # Seção simplificada para frames principais detectados
        st.subheader("Momento crítico detectado")
        
        # Identificando o momento com maior probabilidade para exibir
        max_confidence_index = chart_data['Confiança'].idxmax()
        max_confidence_time = chart_data['Tempo (s)'][max_confidence_index]
        max_confidence_value = chart_data['Confiança'][max_confidence_index]
        
        # Exibindo apenas o frame com maior probabilidade de assalto
        st.image(
            "https://via.placeholder.com/800x450?text=Momento+Crítico",
            caption=f"Momento de maior probabilidade ({max_confidence_value:.2%}) aos {max_confidence_time:.1f}s"
        )
            
    elif submit_button:
        st.warning("Por favor, insira um link válido do YouTube para análise.")

if __name__ == "__main__":
    main()