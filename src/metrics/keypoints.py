import numpy as np
import pandas as pd
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def gerar_relatorio_metricas(metrics_df, output_dir, nome_video=None):
    """Gera relatório com métricas e análises"""
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
            f.write(f"Velocidade média: {metrics_df['velocidade_media'].mean():.4f}\n")
            if 'aceleracao_media' in metrics_df.columns:
                f.write(f"Aceleração média: {metrics_df['aceleracao_media'].mean():.4f}\n")
            if 'jitter_medio' in metrics_df.columns:
                f.write(f"Jitter médio: {metrics_df['jitter_medio'].mean():.4f}\n\n")
        
        # Estabilidade anatômica
        anatomia_cols = [col for col in metrics_df.columns if 'simetria' in col or 'tronco' in col]
        if anatomia_cols:
            f.write("ESTABILIDADE ANATÔMICA:\n")
            for col in anatomia_cols:
                if not metrics_df[col].isna().all():
                    f.write(f"{col}: {metrics_df[col].mean():.4f}\n")
            f.write("\n")
        
        # Conclusão e avaliação
        f.write("CONCLUSÃO:\n")
        qualidade = "Não foi possível avaliar"
        if 'completude_media' in metrics_df.columns and 'confianca_media' in metrics_df.columns:
            completude_avg = metrics_df['completude_media'].mean()
            confianca_avg = metrics_df['confianca_media'].mean()
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
        elif qualidade == "Razoável":
            f.write("- Considerar aplicar suavização temporal\n- Ajustar parâmetros de detecção\n")
        else:
            f.write("- Detecção adequada para a maioria das aplicações\n")
    
    print(f"Relatório de qualidade salvo em: {report_path}")
    return report_path

def calcular_metricas_avancadas(all_keypoints, smoothed_keypoints=None, confidence_scores=None):
    """Calcula métricas avançadas para a sequência de keypoints"""
    result_data = []
    
    # Usar os keypoints suavizados se disponíveis, senão usa os originais
    keypoints_to_analyze = smoothed_keypoints if smoothed_keypoints is not None else all_keypoints
    
    for frame_idx, keypoints in enumerate(keypoints_to_analyze):
        # Inicializa com valores padrão
        metrics = {
            'frame': frame_idx,
            'velocidade_media': 0.0,
            'aceleracao_media': 0.0,
            'fluidez_movimento': 0.0,
            'confianca_media': 0.0
        }
        
        # Se não houver keypoints, adicione os valores padrão
        if keypoints is None or len(keypoints) == 0:
            result_data.append(metrics)
            continue
            
        # Se for um array 3D (múltiplas pessoas), use apenas o primeiro conjunto
        if len(keypoints.shape) == 3:
            keypoints = keypoints[0]
            
        # Calcula confiança média, se disponível
        if confidence_scores is not None and frame_idx < len(confidence_scores) and confidence_scores[frame_idx] is not None:
            conf_scores = confidence_scores[frame_idx]
            # Verifica se os scores são para várias partes do corpo
            if isinstance(conf_scores, np.ndarray) and conf_scores.size > 1:
                metrics['confianca_media'] = np.nanmean(conf_scores)
            elif isinstance(conf_scores, (float, int)):
                metrics['confianca_media'] = float(conf_scores)
        
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
                        
                        # Calcula aceleração usando a diferença de velocidades
                        vel_atual = np.sqrt(np.nansum((keypoints - prev_keypoints)**2, axis=1))
                        vel_prox = np.sqrt(np.nansum((next_keypoints - keypoints)**2, axis=1))
                        aceleracoes = np.abs(vel_prox - vel_atual)
                        
                        metrics['aceleracao_media'] = np.nanmean(aceleracoes)
                        
                        # Fluidez como inverso da variância das acelerações
                        if len(aceleracoes) > 0:
                            metrics['fluidez_movimento'] = 1.0 / (1.0 + np.nanvar(aceleracoes))
        
        result_data.append(metrics)
    
    return pd.DataFrame(result_data)

def calcular_metricas_avancadas(all_keypoints, smoothed_keypoints=None, confidence_scores=None):
    """Calcula métricas avançadas para a sequência de keypoints"""
    result_data = []
    
    # Usar os keypoints suavizados se disponíveis, senão usa os originais
    keypoints_to_analyze = smoothed_keypoints if smoothed_keypoints is not None else all_keypoints
    
    for frame_idx, keypoints in enumerate(keypoints_to_analyze):
        # Inicializa com valores padrão
        metrics = {
            'frame': frame_idx,
            'velocidade_media': 0.0,
            'aceleracao_media': 0.0,
            'fluidez_movimento': 0.0,
            'confianca_media': 0.0,
            'confianca_min': 0.0,  # Adicionado valor padrão para confianca_min
            'confianca_std': 0.0   # Também podemos adicionar o desvio padrão
        }
        
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
                metrics['confianca_min'] = np.nanmin(conf_scores)  # Calcula o mínimo
                metrics['confianca_std'] = np.nanstd(conf_scores)   # Calcula o desvio padrão
            elif isinstance(conf_scores, (float, int)):
                metrics['confianca_media'] = float(conf_scores)
                metrics['confianca_min'] = float(conf_scores)  # Se houver apenas um valor, mínimo = média
                metrics['confianca_std'] = 0.0                 # Desvio padrão zero para um único valor
        
        # [resto da função permanece igual]
        
        result_data.append(metrics)
    
    return pd.DataFrame(result_data)

def calcular_estabilidade_anatomica(all_keypoints, pose_connections):
    """Calcula métricas de estabilidade anatômica para sequência de keypoints"""
    result_data = []
    
    for frame_idx, keypoints in enumerate(all_keypoints):
        # Se não houver keypoints, adicione uma entrada vazia
        if keypoints is None or len(keypoints) == 0:
            result_data.append({
                'frame': frame_idx,
                'proporcao_valida': 0.0,
                'simetria_corporal': 0.0
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
        
        result_data.append({
            'frame': frame_idx,
            'proporcao_valida': proporcao_valida,
            'simetria_corporal': simetria
        })
    
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

    