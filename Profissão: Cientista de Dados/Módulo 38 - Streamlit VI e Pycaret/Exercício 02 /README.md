# Construção de um Modelo de Credit Scoring para Cartão de Crédito
#### (usando PYCARET)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Machine Learning](https://img.shields.io/badge/Machine_Learning-F39C12?style=for-the-badge)](https://en.wikipedia.org/wiki/Machine_learning)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Seaborn](https://img.shields.io/badge/Seaborn-4EAE4E?style=for-the-badge&logo=seaborn&logoColor=white)](https://seaborn.pydata.org/)
[![PyCaret](https://img.shields.io/badge/PyCaret-FF8000?style=for-the-badge&logo=pycaret&logoColor=white)](https://pycaret.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3776AB?style=for-the-badge&logo=matplotlib&logoColor=white)](https://matplotlib.org/)

## Descrição do Projeto

Este projeto tem como objetivo desenvolver um modelo de Credit Scoring para avaliação de risco de crédito em solicitações de cartão de crédito. Utilizando dados amostrais provenientes de 15 safras, com informações referentes a 12 meses de desempenho, a abordagem adotada visa fornecer uma análise preditiva robusta.

## Pré-processamento de Dados

O conjunto de dados é explorado e pré-processado utilizando ferramentas como Streamlit e PyCaret. O Streamlit oferece uma interface interativa para visualização e seleção de arquivos, enquanto o PyCaret simplifica tarefas essenciais como a configuração do experimento, seleção de modelos, ajuste de hiperparâmetros e avaliação de desempenho.

## Transformação de Dados

Ao longo do desenvolvimento, são realizadas transformações nos dados, removendo colunas desnecessárias e forçando a conversão de variáveis importantes. A matriz de correlação é examinada para entender as relações entre as variáveis, e a contagem de valores únicos proporciona uma visão inicial da distribuição do target.

## Bibliotecas Utilizadas

```python
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pycaret.classification import setup, models, create_model, tune_model, plot_model, evaluate_model, finalize_model, predict_model
import os
import pickle
```

## Base de Dados

A base de dados utilizada neste projeto é `credit_scoring.ftr`.

## Configuração do Experimento

O projeto também abrange a configuração do experimento utilizando o PyCaret, a seleção e treinamento de um modelo de machine learning (LightGBM), e a avaliação do desempenho do modelo com métricas como a Curva ROC, Matriz de Confusão, e a Curva Precision-Recall.

## Resultados

Por fim, a aplicação Streamlit resultante permite a visualização interativa dos resultados do modelo treinado, incluindo gráficos de desempenho e insights sobre a importância das features utilizadas. O código está estruturado de forma a possibilitar futuras iterações e ajustes no modelo à medida que novos dados forem disponibilizados.

[projetoModulo38tarefa2.webm](https://github.com/Aline-Castro/Ciencia-de-Dados-EBAC/assets/92234598/15e973ec-a43a-4c5c-a6f4-851ac2cf4d0a)
