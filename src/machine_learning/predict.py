"""
Script de Predição para Doença Cardíaca
Carrega o modelo treinado e faz predições sobre novos dados
"""

import joblib
import numpy as np
import pandas as pd
from typing import Union, List, Dict


class HeartDiseasePredictor:
    """Classe para fazer predições de doença cardíaca"""
    
    def __init__(self, model_path='heart_disease_model.pkl', 
                 scaler_path='scaler.pkl',
                 feature_names_path='feature_names.pkl'):
        """
        Inicializa o preditor carregando o modelo e scaler
        
        Args:
            model_path: Caminho para o ficheiro do modelo
            scaler_path: Caminho para o ficheiro do scaler
            feature_names_path: Caminho para os nomes das features
        """
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_names = joblib.load(feature_names_path)
        
        print(f"Modelo carregado: {type(self.model).__name__}")
        print(f"Features esperadas ({len(self.feature_names)}): {self.feature_names}")
    
    def predict(self, features: Union[np.ndarray, pd.DataFrame, List, Dict]) -> Dict:
        """
        Faz predição sobre as features fornecidas
        
        Args:
            features: Features do paciente (array, DataFrame, lista ou dicionário)
        
        Returns:
            Dicionário com a predição e probabilidade
        """
        # Converter para DataFrame se necessário
        if isinstance(features, dict):
            features_df = pd.DataFrame([features])
        elif isinstance(features, list):
            features_df = pd.DataFrame([features], columns=self.feature_names)
        elif isinstance(features, np.ndarray):
            if features.ndim == 1:
                features = features.reshape(1, -1)
            features_df = pd.DataFrame(features, columns=self.feature_names)
        elif isinstance(features, pd.DataFrame):
            features_df = features
        else:
            raise ValueError("Formato de features não suportado")
        
        # Validar que temos todas as features necessárias
        if not all(col in features_df.columns for col in self.feature_names):
            missing = [col for col in self.feature_names if col not in features_df.columns]
            raise ValueError(f"Features em falta: {missing}")
        
        # Ordenar colunas na ordem correta
        features_df = features_df[self.feature_names]
        
        # Normalizar
        features_scaled = self.scaler.transform(features_df)
        
        # Predição
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        # Resultado
        result = {
            'prediction': int(prediction),
            'resultado': 'TEM doença cardíaca' if prediction == 1 else 'NÃO TEM doença cardíaca',
            'probabilidade_sem_doenca': float(probability[0]),
            'probabilidade_com_doenca': float(probability[1]),
            'confianca': float(max(probability))
        }
        
        return result
    
    def predict_batch(self, features_list: Union[pd.DataFrame, List[Dict]]) -> List[Dict]:
        """
        Faz predições em lote
        
        Args:
            features_list: Lista de features ou DataFrame
        
        Returns:
            Lista de dicionários com predições
        """
        if isinstance(features_list, pd.DataFrame):
            return [self.predict(row.to_dict()) for _, row in features_list.iterrows()]
        else:
            return [self.predict(features) for features in features_list]


def main():
    """Função principal com exemplos de uso"""
    
    # Carregar o preditor
    predictor = HeartDiseasePredictor()
    
    print("\n" + "="*70)
    print("EXEMPLO DE USO DO PREDITOR")
    print("="*70)
    
    # Exemplo 1: Usando dicionário (mais fácil de ler)
    print("\n1. Predição com dicionário:")
    paciente1 = {
        'age': 63,
        'sex': 1,
        'cp': 3,
        'trestbps': 145,
        'chol': 233,
        'fbs': 1,
        'restecg': 0,
        'thalach': 150,
        'exang': 0,
        'oldpeak': 2.3,
        'slope': 0,
        'ca': 0,
        'thal': 1
    }
    
    resultado1 = predictor.predict(paciente1)
    print(f"Resultado: {resultado1['resultado']}")
    print(f"Probabilidade de doença: {resultado1['probabilidade_com_doenca']:.2%}")
    print(f"Confiança: {resultado1['confianca']:.2%}")
    
    # Exemplo 2: Usando lista (na ordem das features)
    print("\n2. Predição com lista:")
    paciente2 = [67, 1, 0, 160, 286, 0, 0, 108, 1, 1.5, 1, 3, 2]
    
    resultado2 = predictor.predict(paciente2)
    print(f"Resultado: {resultado2['resultado']}")
    print(f"Probabilidade de doença: {resultado2['probabilidade_com_doenca']:.2%}")
    print(f"Confiança: {resultado2['confianca']:.2%}")
    
    # Exemplo 3: Predição em lote
    print("\n3. Predição em lote (múltiplos pacientes):")
    pacientes = [
        {'age': 54, 'sex': 1, 'cp': 0, 'trestbps': 140, 'chol': 239, 'fbs': 0, 
         'restecg': 1, 'thalach': 160, 'exang': 0, 'oldpeak': 1.2, 'slope': 0, 'ca': 0, 'thal': 2},
        {'age': 41, 'sex': 0, 'cp': 1, 'trestbps': 130, 'chol': 204, 'fbs': 0, 
         'restecg': 0, 'thalach': 172, 'exang': 0, 'oldpeak': 1.4, 'slope': 2, 'ca': 0, 'thal': 2},
    ]
    
    resultados = predictor.predict_batch(pacientes)
    for i, res in enumerate(resultados, 1):
        print(f"\nPaciente {i}: {res['resultado']} (Prob: {res['probabilidade_com_doenca']:.2%})")
    
    print("\n" + "="*70)
    print("Para usar este script:")
    print("1. Importe a classe: from predict import HeartDiseasePredictor")
    print("2. Crie uma instância: predictor = HeartDiseasePredictor()")
    print("3. Faça predições: resultado = predictor.predict(features)")
    print("="*70)


if __name__ == "__main__":
    main()
