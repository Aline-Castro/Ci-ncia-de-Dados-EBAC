# Importando as bibliotecas necessárias:
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

def main():
    # Configurando a página do Streamlit
    st.set_page_config(page_title='Análise de Telemarketing', 
                       page_icon='D:\\EBAC\\CIENCIA DE DADOS\\19\\Nova pasta\\img\\telmarketing_icon.png',
                       layout="wide",
                       initial_sidebar_state='expanded')
    
    # Escrevendo o título da página
    st.write('# Análise de Telemarketing')
    st.markdown("---")

    # Carregando e exibindo a imagem na barra lateral
    image = Image.open("D:\\EBAC\\CIENCIA DE DADOS\\19\\Nova pasta\\img\\Bank-Branding.jpg")
    st.sidebar.image(image)

    # Carregando a base de dados:
    bank_raw = pd.read_csv('D:\\EBAC\\CIENCIA DE DADOS\\19\\Nova pasta\\data\\input\\bank-additional-full.csv', delimiter=';')
    bank = bank_raw.copy()

    # Exibindo os dados brutos antes de qualquer filtro
    st.write('## Antes dos filtros')
    st.write(bank_raw.head())

    # Selecionando a faixa de idades com um controle deslizante
    max_age = int(bank.age.max())
    min_age = int(bank.age.min())
    idades = st.sidebar.slider(label='Idade', 
                        min_value = min_age,
                        max_value = max_age, 
                        value = (min_age, max_age),
                        step = 1)
    st.sidebar.write('IDADES:', idades)
    st.sidebar.write('IDADE MÍNIMA:', idades[0])
    st.sidebar.write('IDADE MÁXIMA:', idades[1])

    # Selecionando as profissões com uma lista de seleção múltipla
    jobs_list = bank.job.unique().tolist()
    st.sidebar.write('PROFISSÕES DISPONÍVEIS:', jobs_list)
    jobs_selected =  st.sidebar.multiselect("Profissão", jobs_list, jobs_list)
    st.sidebar.write('PROFISSÕES SELECIONADAS:', jobs_selected)

    # Filtrando os dados de acordo com as idades e profissões selecionadas
    bank = bank[(bank['age'] >= idades[0]) & (bank['age'] <= idades[1])]
    bank = bank[bank['job'].isin(jobs_selected)].reset_index(drop=True)

    # Exibindo os dados após os filtros
    st.write('## Após os filtros')
    st.write(bank.head())
    st.markdown("---")

    # Criando gráficos para comparar os dados brutos e filtrados
    fig, ax = plt.subplots(1, 2, figsize = (5,3))

    # Gráfico dos dados brutos
    bank_raw.y.value_counts(normalize=True).plot(kind='bar', ax=ax[0], color=['steelblue', 'darksalmon'])
    ax[0].set_title('Dados brutos', fontweight ="bold")
    ax[0].set_xlabel('Resposta')
    ax[0].set_ylabel('Proporção')
    
    # Gráfico dos dados filtrados
    bank.y.value_counts(normalize=True).plot(kind='bar', ax=ax[1], color=['steelblue', 'darksalmon'])
    ax[1].set_title('Dados filtrados', fontweight ="bold")
    ax[1].set_xlabel('Resposta')
    #ax[1].set_ylabel('Proporção')
    
    # Exibindo os gráficos
    st.write('## Proporção de aceite')
    st.pyplot(fig)

# Executando a função principal
if __name__ == '__main__':
    main()
