# Projeto de Machine Learning - Predição de Doença Cardíaca

## 📋 Descrição

Este projeto utiliza o dataset de doenças cardíacas do UCI Machine Learning Repository para treinar um modelo de classificação que prevê a presença de doença cardíaca em pacientes.

## 🗂️ Estrutura do Projeto

```
iacd_proj/
├── ml.ipynb                      # Notebook com todo o pipeline de ML
├── predict.py                    # Script de predição
├── heart_disease_model.pkl       # Modelo treinado (gerado após execução)
├── scaler.pkl                    # Scaler para normalização (gerado após execução)
├── feature_names.pkl             # Nomes das features (gerado após execução)
└── README_ML.md                  # Este ficheiro
```

## 🚀 Como Usar

### 1. Executar o Notebook

Abra `ml.ipynb` e execute todas as células sequencialmente. O notebook irá:

1. **Carregar os dados** do UCI Repository
2. **Explorar e limpar** os dados (tratar missing values)
3. **Dividir** em treino/teste (80/20)
4. **Treinar dois modelos**:
   - Random Forest
   - Logistic Regression
5. **Avaliar** os modelos com métricas:
   - Acurácia
   - ROC AUC
   - Classification Report
   - Confusion Matrix
6. **Guardar** o melhor modelo e o scaler

### 2. Fazer Predições

Depois de executar o notebook, use o script `predict.py`:

```python
from predict import HeartDiseasePredictor

# Criar preditor
predictor = HeartDiseasePredictor()

# Fazer predição com dicionário
paciente = {
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

resultado = predictor.predict(paciente)
print(resultado['resultado'])  # "TEM doença cardíaca" ou "NÃO TEM doença cardíaca"
print(f"Probabilidade: {resultado['probabilidade_com_doenca']:.2%}")
```

### 3. Executar o Script de Exemplo

```powershell
.\.venv\Scripts\Activate.ps1
python predict.py
```

Isso mostrará exemplos de predições e como usar o preditor.

## 📊 Features do Dataset

O modelo usa as seguintes 13 features:

1. **age**: Idade do paciente
2. **sex**: Sexo (1 = masculino, 0 = feminino)
3. **cp**: Tipo de dor no peito (0-3)
4. **trestbps**: Pressão arterial em repouso (mm Hg)
5. **chol**: Colesterol sérico (mg/dl)
6. **fbs**: Açúcar no sangue em jejum > 120 mg/dl (1 = sim, 0 = não)
7. **restecg**: Resultados do eletrocardiograma em repouso (0-2)
8. **thalach**: Frequência cardíaca máxima atingida
9. **exang**: Angina induzida por exercício (1 = sim, 0 = não)
10. **oldpeak**: Depressão do ST induzida por exercício
11. **slope**: Inclinação do segmento ST do pico do exercício (0-2)
12. **ca**: Número de vasos principais coloridos por fluoroscopia (0-3)
13. **thal**: Thalassemia (0 = normal, 1 = defeito fixo, 2 = defeito reversível)

## 📈 Resultados Esperados

- **Acurácia**: ~80-85%
- **ROC AUC**: ~0.85-0.90

Os resultados exatos variam dependendo da limpeza dos dados e do modelo selecionado.

## 🔧 Dependências

```python
pandas
numpy
scikit-learn
joblib
ucimlrepo
```

## 💡 Notas

- O modelo converte o target original (0-4) para binário (0 = sem doença, 1 = com doença)
- Os dados são normalizados usando `StandardScaler`
- O melhor modelo é selecionado automaticamente baseado no ROC AUC
- Todos os ficheiros necessários para predição são guardados automaticamente

## 🎯 Próximos Passos Possíveis

- [ ] Testar outros algoritmos (XGBoost, SVM, etc.)
- [ ] Fazer grid search para otimizar hiperparâmetros
- [ ] Adicionar validação cruzada
- [ ] Criar visualizações (ROC curve, feature importance)
- [ ] Implementar tratamento mais sofisticado de missing values
- [ ] Adicionar interface web para predições
