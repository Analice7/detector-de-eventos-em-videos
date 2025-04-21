import numpy as np

# Imports para DeepSORT
import torch
from .features import extract_features

class DeepSORTKeypointTracker:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3, feature_weight=0.7):
        """
        Inicializa o rastreador DeepSORT adaptado para keypoints com features avançadas.
        
        Args:
            max_age: Número máximo de frames para manter um rastro sem atualização
            min_hits: Número mínimo de detecções para considerar um rastro confirmado
            iou_threshold: Limiar IoU para associar detecções a rastros
            feature_weight: Peso para similaridade de features (0-1), onde 1 significa apenas features
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.feature_weight = feature_weight
        self.next_id = 0
        self.tracks = []  # Lista de rastros ativos
        self.frame_count = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"DeepSORT usando device: {self.device}")
        
        # Manter histórico de keypoints para extrair características cinemáticas
        self.keypoints_history = {}  # Dicionário para armazenar histórico de keypoints por ID
        self.history_size = 10  # Número de frames para manter no histórico
        
    def _get_keypoint_bbox(self, keypoints):
        """Extrai bounding box dos keypoints de uma pessoa"""
        if keypoints is None or len(keypoints) == 0:
            return None
        
        # Filtrar keypoints válidos (não NaN)
        valid_kps = keypoints[~np.isnan(keypoints).any(axis=1)]
        if len(valid_kps) == 0:
            return None
        
        # Calcular bounding box dos keypoints
        x_min, y_min = valid_kps.min(axis=0)
        x_max, y_max = valid_kps.max(axis=0)
        
        # Formato [x1, y1, x2, y2, confiança]
        return np.array([x_min, y_min, x_max, y_max, 1.0])
    
    def _calculate_iou(self, bbox1, bbox2):
        """Calcula IoU entre duas bounding boxes no formato [x1, y1, x2, y2]"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        # Verificar se há sobreposição
        if x2 < x1 or y2 < y1:
            return 0.0
        
        # Calcular áreas
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_feature_similarity(self, feature1, feature2):
        """
        Calcula a similaridade entre dois dicionários de features.
        Retorna valor entre 0 (diferente) e 1 (idêntico).
        """
        if not feature1 or not feature2:
            return 0.0
            
        # Obter características comuns aos dois dicionários
        common_keys = set(feature1.keys()).intersection(set(feature2.keys()))
        if not common_keys:
            return 0.0
        
        # Calcular similaridade para cada característica
        similarities = []
        for key in common_keys:
            try:
                # Converter para arrays numpy se já não forem
                f1 = np.asarray(feature1[key], dtype=np.float32)
                f2 = np.asarray(feature2[key], dtype=np.float32)
                
                # Usar o último valor de cada característica para comparação
                # (assumindo que são séries temporais)
                val1 = f1[-1] if f1.size > 0 else 0
                val2 = f2[-1] if f2.size > 0 else 0
                
                # Normalizar os valores para [0, 1] com base em um intervalo típico
                # Isso depende do tipo de característica
                if "angle" in key:
                    # Ângulos podem variar de 0 a 180 graus
                    diff = abs(val1 - val2) / 180.0
                elif "vel" in key or "acc" in key:
                    # Usar uma escala relativa para velocidades e acelerações
                    max_val = max(abs(val1), abs(val2))
                    diff = abs(val1 - val2) / (max_val + 1e-6)
                else:
                    # Para posições, usar uma escala relativa
                    max_val = max(abs(val1), abs(val2))
                    diff = abs(val1 - val2) / (max_val + 1e-6)
                
                # Converter diferença para similaridade (0 = diferente, 1 = idêntico)
                sim = 1.0 - min(1.0, diff)
                similarities.append(sim)
                
            except Exception:
                # Em caso de erro, considerar similaridade média
                similarities.append(0.5)
        
        # Média das similaridades
        if similarities:
            return sum(similarities) / len(similarities)
        else:
            return 0.0
        
    def _extract_features(self, keypoints, track_id=None):
        """
        Extrai características cinemáticas dos keypoints.
        Utiliza o histórico de keypoints para calcular velocidades e acelerações.
        
        Args:
            keypoints: Array de keypoints da pessoa
            track_id: ID do track (para buscar histórico)
            
        Returns:
            Dicionário com as características extraídas
        """
        
        if keypoints is None or len(keypoints) == 0:
            return {}
        
        # Se temos um track_id, devemos usar o histórico para esse track
        if track_id is not None and track_id in self.keypoints_history:
            # Adicionar keypoints atuais ao histórico
            self.keypoints_history[track_id].append(keypoints)
            
            # Manter apenas os últimos N frames
            if len(self.keypoints_history[track_id]) > self.history_size:
                self.keypoints_history[track_id] = self.keypoints_history[track_id][-self.history_size:]
            
            # Extrair características do histórico completo
            keypoints_sequence = np.array(self.keypoints_history[track_id])
            return extract_features(keypoints_sequence, feature_type="all", normalize=True)
        else:
            # Se não temos histórico, criamos um array de sequência com apenas o keypoint atual
            keypoints_sequence = np.array([keypoints])
            # Neste caso só extraímos características de posição, já que não temos histórico para velocidade/aceleração
            return extract_features(keypoints_sequence, feature_type="position", normalize=True)
    
    def _associate_detections_to_tracks(self, detections, frame=None):
        """
        Associa novas detecções a rastros existentes usando combinação de IoU e features.
        
        Args:
            detections: Lista de keypoints detectados
            frame: Frame atual para extração de features (opcional)
            
        Returns:
            matches, unmatched_tracks, unmatched_detections
        """
        if len(self.tracks) == 0 or len(detections) == 0:
            return [], list(range(len(self.tracks))), list(range(len(detections)))
        
        # Matriz de custos combinando IoU e features
        cost_matrix = np.zeros((len(self.tracks), len(detections)))
        
        # Para cada par track-detecção, calcular custo combinado
        for t, track in enumerate(self.tracks):
            track_bbox = self._get_keypoint_bbox(track['keypoints'])
            track_features = track.get('features')
            track_id = track.get('id')
            
            if track_bbox is None:
                cost_matrix[t, :] = 1.0  # Custo máximo
                continue
                
            for d, detection in enumerate(detections):
                det_bbox = self._get_keypoint_bbox(detection)
                
                if det_bbox is None:
                    cost_matrix[t, d] = 1.0  # Custo máximo
                    continue
                
                # 1. Componente IoU
                iou = self._calculate_iou(track_bbox, det_bbox)
                iou_cost = 1.0 - iou
                
                # 2. Componente de similaridade de features
                feature_cost = 1.0  # Valor padrão (custo máximo)
                
                # Extrair features do keypoint atual (sem histórico)
                det_features = self._extract_features(detection)
                
                if track_features and det_features:
                    # Calcular similaridade (0=diferente, 1=igual)
                    similarity = self._calculate_feature_similarity(track_features, det_features)
                    # Converter para custo (0=igual, 1=diferente)
                    feature_cost = 1.0 - similarity
                
                # Combinar custos com o peso especificado
                # feature_weight determina a importância relativa das features vs IoU
                combined_cost = (self.feature_weight * feature_cost + 
                                (1.0 - self.feature_weight) * iou_cost)
                
                cost_matrix[t, d] = combined_cost
        
        # Associação gulosa (greedy matching)
        matches = []
        unmatched_tracks = list(range(len(self.tracks)))
        unmatched_detections = list(range(len(detections)))
        
        # Para cada track, encontrar a melhor detecção disponível
        for t in unmatched_tracks.copy():
            costs = [(d, cost_matrix[t, d]) for d in unmatched_detections]
            costs.sort(key=lambda x: x[1])  # Ordenar por custo (menor primeiro)
            
            if costs and costs[0][1] < 0.7:  # Limiar de custo máximo para considerar um match
                best_det_idx = costs[0][0]
                matches.append((t, best_det_idx))
                unmatched_tracks.remove(t)
                unmatched_detections.remove(best_det_idx)
        
        return matches, unmatched_tracks, unmatched_detections
    
    def update(self, keypoints_list, frame=None):
        """
        Atualiza o rastreador com novas detecções.
        
        Args:
            keypoints_list: Lista de arrays de keypoints detectados no frame atual
            frame: Frame atual (não utilizado nesta implementação focada em características cinemáticas)
            
        Returns:
            Lista de keypoints organizados por ID consistente
        """
        self.frame_count += 1
        
        # Filtrar keypoints não vazios
        filtered_keypoints = [kp for kp in keypoints_list if kp is not None and len(kp) > 0]
        
        # Associar detecções a tracks existentes
        matches, unmatched_tracks, unmatched_detections = self._associate_detections_to_tracks(filtered_keypoints)
        
        # Atualizar tracks associados
        for track_idx, detection_idx in matches:
            track_id = self.tracks[track_idx]['id']
            self.tracks[track_idx]['keypoints'] = filtered_keypoints[detection_idx]
            
            # Atualizar histórico e extrair características considerando o histórico
            if track_id not in self.keypoints_history:
                self.keypoints_history[track_id] = []
            
            # Atualizar features do track com base no histórico
            self.tracks[track_idx]['features'] = self._extract_features(
                filtered_keypoints[detection_idx], track_id=track_id
            )
            
            self.tracks[track_idx]['age'] = 0
            self.tracks[track_idx]['hits'] += 1
            
        # Atualizar tracks não associados
        for track_idx in unmatched_tracks:
            self.tracks[track_idx]['age'] += 1
            
        # Iniciar novos tracks para detecções não associadas
        for detection_idx in unmatched_detections:
            new_id = self.next_id
            
            # Inicializar histórico para o novo track
            self.keypoints_history[new_id] = [filtered_keypoints[detection_idx]]
            
            # Inicializar features apenas com características de posição
            initial_features = self._extract_features(filtered_keypoints[detection_idx])
            
            self.tracks.append({
                'id': new_id,
                'keypoints': filtered_keypoints[detection_idx],
                'features': initial_features,
                'age': 0,
                'hits': 1
            })
            self.next_id += 1
            
        # Remover tracks antigos
        old_track_ids = [track['id'] for track in self.tracks if track['age'] > self.max_age]
        self.tracks = [track for track in self.tracks if track['age'] <= self.max_age]
        
        # Limpar histórico de tracks removidos
        for old_id in old_track_ids:
            if old_id in self.keypoints_history:
                del self.keypoints_history[old_id]
        
        # Organizar e retornar keypoints por ID consistente
        active_tracks = [track for track in self.tracks if track['hits'] >= self.min_hits]
        sorted_tracks = sorted(active_tracks, key=lambda t: t['id'])
        
        # Criar lista ordenada de keypoints, usando o mesmo formato do array original
        tracked_keypoints = [track['keypoints'] for track in sorted_tracks]
        
        return tracked_keypoints