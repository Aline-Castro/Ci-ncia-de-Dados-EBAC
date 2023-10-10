# Importando as bibliotecas necessárias
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time

st.set_page_config(
    page_title="EBAC | Módulo 15 | Streamlit I | Exercício 1",
    # page_icon="https://ebaconline.com.br/favicon.ico",
    page_icon="https://raw.githubusercontent.com/rhatiro/Curso_EBAC-Profissao_Cientista_de_Dados/main/ebac-course-utils/media/icon/favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown('''
<img src="https://raw.githubusercontent.com/rhatiro/Curso_EBAC-Profissao_Cientista_de_Dados/main/ebac-course-utils/media/logo/newebac_logo_black_half.png" alt="ebac-logo">

---

# **Profissão: Cientista de Dados**
### **Módulo 15** | Streamlit I | Exercício 1

Aline de Castro Santos<br>
Data: Outubro de 2023.

---
            ''', unsafe_allow_html=True)

st.title('Viagens de Uber em NYC')

DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
            'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

@st.cache_data
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data

data_load_state = st.text('Carregando dados...')
data = load_data(10000)
data_load_state.text("Pronto! (usando st.cache_data)")

if st.checkbox('Mostrar dados brutos'):
    st.subheader('Dados brutos')
    st.write(data)

st.subheader('Número de viagens por hora')
hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
st.bar_chart(hist_values)

# Algum número no intervalo 0-23
hour_to_filter = st.slider('hora', 0, 23, 17)
filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]

st.subheader('Mapa de todas as viagens de Uber às %s:00' % hour_to_filter)
st.map(filtered_data)

# Análise de tendências ao longo do tempo
if st.checkbox('Mostrar tendências ao longo do tempo'):
    st.subheader('Número de viagens por dia')
    hist_values = np.histogram(data[DATE_COLUMN].dt.day, bins=30, range=(1,31))[0]
    st.bar_chart(hist_values)

# Análise geográfica mais detalhada
if st.checkbox('Mostrar análise geográfica detalhada'):
    st.subheader('Mapa de todas as viagens')
    st.map(data)
