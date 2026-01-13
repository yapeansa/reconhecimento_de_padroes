# Trabalho de RP

## Descrição

- Escolha um **tema/problema** atual
- Tipo: **classificação** (recomendado), **regressão** (podendo ser série temporal) ou **agrupamento** (menos indicado, por conta de comparação).
- **Levantamento bibliográfico** sucinto (estado da arte + 3-5 referências relevantes).
- **Apresentação do tema**: definições, 2-3 exemplos de bases/datasets e soluções típicas.
- **Dados para experimentos**: **2 datasets obrigatórios** (opcional: datasets adicionais).
- **Execução de algoritmos da literatura**.
- **Comitês (ensembles)** de votação e ponderação.
- (Opcional) **Implementar** um algoritmo/variação simples (podendo ser algum comportamento aleatório).
- **Metodologia adequada**: preparação/limpeza/balanceamento, split estratificado, métricas corretas, *seed* fixo.
- **Comparação crítica** dos resultados (pontos positivos/negativos).
- **Conclusões**: o que funciona, limites e próximos passos.

---

## Etapas

### Etapa 1 &ndash; **Apenas Algoritmos Clássicos** (Baselines)

**Objetivo**: estabelecer baselines sólidos e baratos.

**Modelos (Escolha de pelo menos 3)**:

- **Regressão Logística**
- **Árvore de Decisão**
- **Floresta Aleatória**
- **SVM** (linear ou outro kernel)
- **Naive Bayes**
- **XGBoost/LightGBM**
- Outro classificador
- Métodos adequados para **regressão** ou **agrupamento**

**Métricas mínimas**:

- **Classificação**: Acurácia, precisão, revocação, F-scores, AUC-ROC, matriz de confusão, outras.
- **Regressão**: MAE e MSE (+ R^{2}).
- **Agrupamento**: Silhouette.

**Protocolo mínimo**:

- **k-fold estratificado**.
- **seed fixo** e relato das versões das libs.
- Sem leakage (ex.: sem normalizar/balancear com informações do teste).

---

### Etapa 2 &ndash; **Incluir um modelo GPT** (e comparar os baselines)

**Objetivo**: medir o ganho/perda de usar GPT vs. clássicos; manter paridade de avaliação.

**Abordagem**:

**Classificação via prompting.**

- **Zero-shot** (instruções + rótulos possíveis).
- **Few-shot** com exemplos do treino/val.

**(Opcional)** Embeddings: sentence embeddings + classificador raso (LR/kNN).

**Métricas (iguais às da Etapa 1) + leves extras**:

- **Mesmos subconjuntos** da Etapa 1.
- **Documentar o prompt e versão do modelo**.
- Mapear saídas do GPT &longrightarrow; **rótulos válidos**.

---
