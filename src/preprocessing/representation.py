import numpy as np

def calculate_velocity(keypoints_data, time_step=1):
    """
    Calcula a velocidade dos keypoints (primeira derivada da posição).
    
    Parâmetros:
    keypoints_data -- array numpy com formato (frames, keypoints, dimensions)
    time_step -- intervalo de tempo entre os frames (padrão: 1)
    
    Retorna:
    Um array com as velocidades dos keypoints com formato (frames-1, keypoints, dimensions)
    """
    # Velocidade é a diferença de posições dividida pelo intervalo de tempo
    velocity = np.diff(keypoints_data, axis=0) / time_step
    
    return velocity

def calculate_acceleration(keypoints_data, time_step=1):
    """
    Calcula a aceleração dos keypoints (segunda derivada da posição).
    
    Parâmetros:
    keypoints_data -- array numpy com formato (frames, keypoints, dimensions)
    time_step -- intervalo de tempo entre os frames (padrão: 1)
    
    Retorna:
    Um array com as acelerações dos keypoints com formato (frames-2, keypoints, dimensions)
    """
    # Aceleração é a segunda derivada da posição
    # Primeiro calculamos a velocidade
    velocity = calculate_velocity(keypoints_data, time_step)
    
    # Depois a derivada da velocidade (aceleração)
    acceleration = np.diff(velocity, axis=0) / time_step
    
    return acceleration
