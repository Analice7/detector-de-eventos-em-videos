import numpy as np
import pandas as pd
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def gerar_relatorio_metricas(metrics_df, output_dir, nome_video=None, fps=None):
    """
    Gera relatório com métricas e análises
    
    Args:
        metrics_df: DataFrame com métricas calculadas
        output_dir: Diretório para salvar o relatório
        nome_video: Nome do vídeo (opcional)
        fps: Frames por segundo do vídeo (opcional)
    """
    output_dir = Path(output_dir)
    if nome_video:
        report_path = output_dir / f"quality_report_{nome_video}.txt"
    else:
        report_path = output_dir / "quality_report.txt"
    
    with open(report_path, 'w') as f:
        titulo = "=== RELATÓRIO DE QUALIDADE DE DETECÇÃO DE KEYPOINTS ==="
        if nome_video:
            titulo += f" - {nome_video}"
        f.write(f"{titulo}\n\n")
        
        # Informações sobre o frame rate
        if fps:
            f.write(f"Frame Rate: {fps} FPS\n\n")
        
        # Estatísticas gerais
        f.write("ESTATÍSTICAS GERAIS:\n")
        f.write(f"Total de frames analisados: {len(metrics_df)}\n")
        if 'total_pessoas' in metrics_df.columns:
            f.write(f"Média de pessoas por frame: {metrics_df['total_pessoas'].mean():.2f}\n")
        f.write("\n")
        
        # Completude
        if 'completude_media' in metrics_df.columns:
            f.write("COMPLETUDE DO ESQUELETO:\n")
            f.write(f"Média: {metrics_df['completude_media'].mean():.2f}%\n")
            f.write(f"Mínima: {metrics_df['completude_min'].min():.2f}%\n")
            f.write(f"Máxima: {metrics_df['completude_max'].max():.2f}%\n\n")
        
        # Confiança
        if 'confianca_media' in metrics_df.columns:
            f.write("CONFIANÇA DA DETECÇÃO:\n")
            f.write(f"Média: {metrics_df['confianca_media'].mean():.4f}\n")
            f.write(f"Mínima: {metrics_df['confianca_min'].min():.4f}\n")
            f.write(f"Desvio padrão: {metrics_df['confianca_std'].mean():.4f}\n\n")
        
        # Métricas temporais
        if 'velocidade_media' in metrics_df.columns:
            f.write("ESTABILIDADE TEMPORAL:\n")
            f.write(f"Velocidade média: {metrics_df['velocidade_media'].mean():.4f}")
            
            # Adiciona informação ajustada pelo frame rate se disponível
            if fps and 'velocidade_media_real' in metrics_df.columns:
                f.write(f" (Ajustada ao tempo real: {metrics_df['velocidade_media_real'].mean():.4f} unidades/s)\n")
            else:
                f.write("\n")
                
            if 'aceleracao_media' in metrics_df.columns:
                f.write(f"Aceleração média: {metrics_df['aceleracao_media'].mean():.4f}")
                
                # Adiciona informação ajustada pelo frame rate se disponível
                if fps and 'aceleracao_media_real' in metrics_df.columns:
                    f.write(f" (Ajustada ao tempo real: {metrics_df['aceleracao_media_real'].mean():.4f} unidades/s²)\n")
                else:
                    f.write("\n")
                    
            if 'jitter_medio' in metrics_df.columns:
                f.write(f"Jitter médio: {metrics_df['jitter_medio'].mean():.4f}")
                
                # Adiciona informação ajustada pelo frame rate se disponível
                if fps and 'jitter_medio_real' in metrics_df.columns:
                    f.write(f" (Ajustado ao tempo real: {metrics_df['jitter_medio_real'].mean():.4f})\n\n")
                else:
                    f.write("\n\n")
        
        # Estabilidade anatômica
        anatomia_cols = [col for col in metrics_df.columns if 'simetria' in col or 'tronco' in col]
        if anatomia_cols:
            f.write("ESTABILIDADE ANATÔMICA:\n")
            for col in anatomia_cols:
                if not metrics_df[col].isna().all():
                    f.write(f"{col}: {metrics_df[col].mean():.4f}\n")
            f.write("\n")
        
        # Análise de Qualidade com base no Frame Rate
        if fps:
            f.write("ANÁLISE DE QUALIDADE BASEADA NO FRAME RATE:\n")
            if fps < 15:
                f.write("Frame rate baixo (<15 FPS): Possível perda de detalhes em movimentos rápidos.\n")
                f.write("Recomendação: Considere interpolar keypoints ou usar modelo otimizado para baixo frame rate.\n\n")
            elif fps >= 15 and fps < 30:
                f.write("Frame rate moderado (15-30 FPS): Adequado para a maioria dos movimentos normais.\n")
                f.write("Recomendação: Monitorar qualidade em seções com movimento rápido.\n\n")
            else:
                f.write("Frame rate alto (>30 FPS): Excelente para capturar movimentos rápidos ou detalhados.\n")
                f.write("Recomendação: Verificar se há sobrecarga computacional desnecessária.\n\n")
        
        # Conclusão e avaliação
        f.write("CONCLUSÃO:\n")
        qualidade = "Não foi possível avaliar"
        if 'completude_media' in metrics_df.columns and 'confianca_media' in metrics_df.columns:
            completude_avg = metrics_df['completude_media'].mean()
            confianca_avg = metrics_df['confianca_media'].mean()
            
            # Ajuste dos limiares com base no frame rate
            if fps:
                # Para FPS baixo, somos um pouco mais tolerantes com a qualidade
                if fps < 15:
                    if completude_avg > 85 and confianca_avg > 0.75:
                        qualidade = "Excelente"
                    elif completude_avg > 75 and confianca_avg > 0.65:
                        qualidade = "Boa"
                    elif completude_avg > 65 and confianca_avg > 0.55:
                        qualidade = "Razoável"
                    else:
                        qualidade = "Baixa"
                # Para FPS alto, podemos ser mais exigentes
                elif fps >= 30:
                    if completude_avg > 92 and confianca_avg > 0.85:
                        qualidade = "Excelente"
                    elif completude_avg > 85 and confianca_avg > 0.75:
                        qualidade = "Boa"
                    elif completude_avg > 75 and confianca_avg > 0.65:
                        qualidade = "Razoável"
                    else:
                        qualidade = "Baixa"
                # Para FPS médio, usamos os limiares originais
                else:
                    if completude_avg > 90 and confianca_avg > 0.8:
                        qualidade = "Excelente"
                    elif completude_avg > 80 and confianca_avg > 0.7:
                        qualidade = "Boa"
                    elif completude_avg > 70 and confianca_avg > 0.6:
                        qualidade = "Razoável"
                    else:
                        qualidade = "Baixa"
            # Se não temos informação sobre FPS, usamos os limiares originais
            else:
                if completude_avg > 90 and confianca_avg > 0.8:
                    qualidade = "Excelente"
                elif completude_avg > 80 and confianca_avg > 0.7:
                    qualidade = "Boa"
                elif completude_avg > 70 and confianca_avg > 0.6:
                    qualidade = "Razoável"
                else:
                    qualidade = "Baixa"
                    
        f.write(f"Qualidade geral da detecção: {qualidade}\n")
        
        # Recomendações
        f.write("\nRECOMENDAÇÕES:\n")
        if qualidade == "Baixa":
            f.write("- Melhorar a iluminação do vídeo\n- Verificar presença de oclusões\n- Usar modelo de pose mais robusto\n")
            if fps and fps < 15:
                f.write("- Considerar aumentar o frame rate para melhorar a detecção\n")
        elif qualidade == "Razoável":
            f.write("- Considerar aplicar suavização temporal\n- Ajustar parâmetros de detecção\n")
            if fps and fps < 20:
                f.write("- Avaliar se o frame rate atual é adequado para o tipo de movimento\n")
        else:
            f.write("- Detecção adequada para a maioria das aplicações\n")
            if fps and fps > 60:
                f.write("- O frame rate atual pode ser excessivo, considere reduzir para economia de processamento\n")
    
    print(f"Relatório de qualidade salvo em: {report_path}")
    return report_path

def calcular_metricas_avancadas(all_keypoints, smoothed_keypoints=None, confidence_scores=None, fps=None):
    """
    Calcula métricas avançadas para a sequência de keypoints
    
    Args:
        all_keypoints: Lista de arrays numpy com keypoints para cada frame
        smoothed_keypoints: Lista de arrays numpy com keypoints suavizados (opcional)
        confidence_scores: Lista de arrays numpy com scores de confiança (opcional)
        fps: Frames por segundo do vídeo (opcional)
    """
    result_data = []
    
    # Usar os keypoints suavizados se disponíveis, senão usa os originais
    keypoints_to_analyze = smoothed_keypoints if smoothed_keypoints is not None else all_keypoints
    
    # Determinar o fator de tempo real (segundos por frame)
    time_factor = 1.0
    if fps is not None and fps > 0:
        time_factor = 1.0 / fps
    
    for frame_idx, keypoints in enumerate(keypoints_to_analyze):
        # Inicializa com valores padrão
        metrics = {
            'frame': frame_idx,
            'velocidade_media': 0.0,
            'aceleracao_media': 0.0,
            'jitter_medio': 0.0,
            'fluidez_movimento': 0.0,
            'confianca_media': 0.0,
            'confianca_min': 0.0,
            'confianca_std': 0.0
        }
        
        # Adicionar métricas ajustadas pelo frame rate
        if fps is not None:
            metrics.update({
                'velocidade_media_real': 0.0,  # unidades/segundo
                'aceleracao_media_real': 0.0,  # unidades/segundo²
                'jitter_medio_real': 0.0       # ajustado pelo tempo
            })
        
        # Se não houver keypoints, adicione os valores padrão
        if keypoints is None or len(keypoints) == 0:
            result_data.append(metrics)
            continue
            
        # Se for um array 3D (múltiplas pessoas), use apenas o primeiro conjunto
        if len(keypoints.shape) == 3:
            keypoints = keypoints[0]
            
        # Calcula confiança média, mínima e desvio padrão, se disponível
        if confidence_scores is not None and frame_idx < len(confidence_scores) and confidence_scores[frame_idx] is not None:
            conf_scores = confidence_scores[frame_idx]
            # Verifica se os scores são para várias partes do corpo
            if isinstance(conf_scores, np.ndarray) and conf_scores.size > 1:
                metrics['confianca_media'] = np.nanmean(conf_scores)
                metrics['confianca_min'] = np.nanmin(conf_scores)
                metrics['confianca_std'] = np.nanstd(conf_scores)
            elif isinstance(conf_scores, (float, int)):
                metrics['confianca_media'] = float(conf_scores)
                metrics['confianca_min'] = float(conf_scores)
                metrics['confianca_std'] = 0.0
        
        # Cálculo de velocidade e aceleração (se houver frames anteriores)
        if frame_idx > 0 and frame_idx < len(keypoints_to_analyze) - 1:
            prev_keypoints = keypoints_to_analyze[frame_idx - 1]
            next_keypoints = keypoints_to_analyze[frame_idx + 1] if frame_idx + 1 < len(keypoints_to_analyze) else None
            
            # Verifica se temos keypoints válidos para calcular velocidade e aceleração
            if (prev_keypoints is not None and next_keypoints is not None and 
                len(prev_keypoints) > 0 and len(next_keypoints) > 0):
                
                # Ensure we're comparing arrays of the same shape
                if isinstance(prev_keypoints, np.ndarray) and isinstance(next_keypoints, np.ndarray):
                    # Lidar com arrays 3D (múltiplas pessoas)
                    if len(prev_keypoints.shape) == 3:
                        prev_keypoints = prev_keypoints[0]
                    if len(next_keypoints.shape) == 3:
                        next_keypoints = next_keypoints[0]
                    
                    # Assegura mesmas dimensões
                    if prev_keypoints.shape == keypoints.shape == next_keypoints.shape:
                        # Calcula velocidade entre frames atual e anterior
                        velocidades = np.sqrt(np.nansum((keypoints - prev_keypoints)**2, axis=1))
                        metrics['velocidade_media'] = np.nanmean(velocidades)
                        
                        # Ajusta para tempo real se FPS fornecido
                        if fps is not None:
                            metrics['velocidade_media_real'] = metrics['velocidade_media'] * fps
                        
                        # Calcula aceleração usando a diferença de velocidades
                        vel_atual = np.sqrt(np.nansum((keypoints - prev_keypoints)**2, axis=1))
                        vel_prox = np.sqrt(np.nansum((next_keypoints - keypoints)**2, axis=1))
                        aceleracoes = np.abs(vel_prox - vel_atual)
                        
                        metrics['aceleracao_media'] = np.nanmean(aceleracoes)
                        
                        # Ajusta para tempo real se FPS fornecido
                        if fps is not None:
                            metrics['aceleracao_media_real'] = metrics['aceleracao_media'] * (fps ** 2)
                        
                        # Fluidez como inverso da variância das acelerações
                        if len(aceleracoes) > 0:
                            metrics['fluidez_movimento'] = 1.0 / (1.0 + np.nanvar(aceleracoes))
                            
                        # Calcula jitter como variação média da posição ajustada pela velocidade
                        # Jitter é maior quando há mudanças bruscas não consistentes com a velocidade geral
                        if frame_idx > 1:
                            prev_prev_keypoints = keypoints_to_analyze[frame_idx - 2]
                            if prev_prev_keypoints is not None and len(prev_prev_keypoints) > 0:
                                if len(prev_prev_keypoints.shape) == 3:
                                    prev_prev_keypoints = prev_prev_keypoints[0]
                                
                                if prev_prev_keypoints.shape == prev_keypoints.shape:
                                    # Calcula diferença entre direções consecutivas (0=suave, valores altos=instável)
                                    dir_prev = prev_keypoints - prev_prev_keypoints
                                    dir_curr = keypoints - prev_keypoints
                                    
                                    # Normaliza direções para comparar apenas ângulos
                                    dir_prev_norm = dir_prev / (np.sqrt(np.nansum(dir_prev**2, axis=1))[:, np.newaxis] + 1e-10)
                                    dir_curr_norm = dir_curr / (np.sqrt(np.nansum(dir_curr**2, axis=1))[:, np.newaxis] + 1e-10)
                                    
                                    # Calcular o produto escalar para obter coseno do ângulo
                                    dot_products = np.nansum(dir_prev_norm * dir_curr_norm, axis=1)
                                    # Limitar para [-1, 1] para evitar erros numéricos
                                    dot_products = np.clip(dot_products, -1.0, 1.0)
                                    # Converter para ângulos (0=mesma direção, pi=direção oposta)
                                    angles = np.arccos(dot_products)
                                    
                                    # Jitter é a média desses ângulos, normalizado para [0,1]
                                    metrics['jitter_medio'] = np.nanmean(angles) / np.pi
                                    
                                    # Ajusta pelo frame rate se disponível
                                    if fps is not None:
                                        # Maior frame rate revela mais jitter
                                        # Normaliza com um fator relacionado ao período de amostragem
                                        metrics['jitter_medio_real'] = metrics['jitter_medio'] * (30.0 / fps if fps > 0 else 1.0)
        
        result_data.append(metrics)
    
    return pd.DataFrame(result_data)

def calcular_estabilidade_anatomica(all_keypoints, pose_connections, fps=None):
    """
    Calcula métricas de estabilidade anatômica para sequência de keypoints
    
    Args:
        all_keypoints: Lista de arrays numpy com keypoints para cada frame
        pose_connections: Lista de pares de índices indicando conexões entre keypoints
        fps: Frames por segundo (opcional)
    """
    result_data = []
    
    for frame_idx, keypoints in enumerate(all_keypoints):
        # Se não houver keypoints, adicione uma entrada vazia
        if keypoints is None or len(keypoints) == 0:
            result_data.append({
                'frame': frame_idx,
                'proporcao_valida': 0.0,
                'simetria_corporal': 0.0,
                'estabilidade_temporal': 0.0 if fps else None
            })
            continue
        
        # Verifique se keypoints é um array 3D (múltiplas pessoas)
        if len(keypoints.shape) == 3:
            # Use apenas a primeira pessoa
            keypoints = keypoints[0]
        
        # Inicializa variáveis
        valid_connections = 0
        total_connections = len(pose_connections)
        left_right_diffs = []
        
        # Calcula proporção válida de conexões
        for c1, c2 in pose_connections:
            if c1 < len(keypoints) and c2 < len(keypoints):
                # Garante que os índices sejam válidos
                if (not np.any(np.isnan(keypoints[c1])) and 
                    not np.any(np.isnan(keypoints[c2]))):
                    valid_connections += 1
        
        proporcao_valida = valid_connections / total_connections if total_connections > 0 else 0
        
        # Calcula simetria corporal (comparando lados esquerdo e direito)
        # Pares de keypoints simétricos (COCO format - ajuste conforme necessário)
        symmetric_pairs = [(5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
        valid_pairs = 0
        
        for left_idx, right_idx in symmetric_pairs:
            if (left_idx < len(keypoints) and right_idx < len(keypoints) and 
                not np.any(np.isnan(keypoints[left_idx])) and 
                not np.any(np.isnan(keypoints[right_idx]))):
                
                # Calcular diferença de posição relativa
                left_pos = keypoints[left_idx]
                right_pos = keypoints[right_idx]
                
                # Diferença relativa à largura da pessoa
                diff = np.abs(left_pos[0] - (1 - right_pos[0]))  # Assumindo coordenadas normalizadas
                left_right_diffs.append(diff)
                valid_pairs += 1
        
        # Calcular média das diferenças (simetria)
        simetria = 1.0 - (np.mean(left_right_diffs) if left_right_diffs else 0)
        
        # Métricas para resultados
        result_metrics = {
            'frame': frame_idx,
            'proporcao_valida': proporcao_valida,
            'simetria_corporal': simetria
        }
        
        # Adiciona medida de estabilidade temporal se tiver pelo menos 3 frames e fps fornecido
        if frame_idx >= 2 and fps:
            # Verifica frames anteriores para medir estabilidade temporal
            prev_frames = [all_keypoints[i] for i in range(max(0, frame_idx-3), frame_idx)]
            valid_prev_frames = [f for f in prev_frames if f is not None and len(f) > 0]
            
            if len(valid_prev_frames) >= 2:
                # Calcula variação da posição ao longo do tempo, normalizada pelo fps
                posicoes = []
                for frame in valid_prev_frames:
                    if len(frame.shape) == 3:
                        frame = frame[0]
                    # Usa o centroide como referência
                    valid_kps = frame[~np.any(np.isnan(frame), axis=1)]
                    if len(valid_kps) > 0:
                        posicoes.append(np.mean(valid_kps, axis=0))
                
                if len(posicoes) >= 2:
                    # Calcula variação média normalizada pelo intervalo de tempo
                    diffs = [np.sqrt(np.sum((posicoes[i+1] - posicoes[i])**2)) for i in range(len(posicoes)-1)]
                    temporal_stability = 1.0 / (1.0 + np.mean(diffs) * fps)
                    result_metrics['estabilidade_temporal'] = temporal_stability
        
        result_data.append(result_metrics)
    
    return pd.DataFrame(result_data)

def gerar_visualizacoes_metricas(metricas_df, output_dir):
    """
    Gera visualizações para as métricas calculadas
    
    Args:
        metricas_df: DataFrame com as métricas
        output_dir: Diretório para salvar as visualizações
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Completude do esqueleto
    if 'completude_media' in metricas_df.columns:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=metricas_df, x='frame', y='completude_media')
        plt.title('Completude Média do Esqueleto ao Longo do Tempo')
        plt.xlabel('Frame')
        plt.ylabel('Completude (%)')
        plt.savefig(output_dir / 'completude_esqueleto.png')
        plt.close()
    
    # 2. Confiança da detecção
    if 'confianca_media' in metricas_df.columns:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=metricas_df, x='frame', y='confianca_media')
        if 'confianca_min' in metricas_df.columns:
            sns.lineplot(data=metricas_df, x='frame', y='confianca_min', alpha=0.5)
        plt.title('Confiança Média da Detecção de Keypoints')
        plt.xlabel('Frame')
        plt.ylabel('Score de Confiança')
        plt.savefig(output_dir / 'confianca_deteccao.png')
        plt.close()
    
    # 3. Velocidade e aceleração
    if 'velocidade_media' in metricas_df.columns:
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        sns.lineplot(data=metricas_df, x='frame', y='velocidade_media')
        plt.title('Velocidade Média do Movimento')
        plt.xlabel('Frame')
        plt.ylabel('Velocidade')
        
        if 'aceleracao_media' in metricas_df.columns:
            plt.subplot(2, 1, 2)
            sns.lineplot(data=metricas_df, x='frame', y='aceleracao_media')
            plt.title('Aceleração Média do Movimento')
            plt.xlabel('Frame')
            plt.ylabel('Aceleração')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'velocidade_aceleracao.png')
        plt.close()
    
    # 4. Jitter (ruído)
    if 'jitter_medio' in metricas_df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=metricas_df, y='jitter_medio')
        plt.title('Distribuição do Jitter (Ruído) dos Keypoints')
        plt.ylabel('Jitter')
        plt.savefig(output_dir / 'jitter_distribuicao.png')
        plt.close()
    
    # 5. Correlação entre métricas
    corr_columns = [col for col in metricas_df.columns if col not in ['frame', 'pessoa']]
    if len(corr_columns) > 1:
        corr_df = metricas_df[corr_columns].corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlação entre Métricas de Qualidade')
        plt.tight_layout()
        plt.savefig(output_dir / 'correlacao_metricas.png')
        plt.close()

    
