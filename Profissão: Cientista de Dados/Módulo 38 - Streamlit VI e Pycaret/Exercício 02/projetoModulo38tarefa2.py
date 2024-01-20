import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pycaret.classification import setup, models, create_model, tune_model, plot_model, evaluate_model, finalize_model, predict_model
import os
import pickle



# Função principal da aplicação
def main():
    # Configuração inicial da página da aplicação
    st.set_page_config(
        page_title="EBAC | Módulo 38 | Tarefa II",
        page_icon='https://raw.githubusercontent.com/Aline-Castro/RFV/main/favicon.ico', 
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.markdown('''
                        <div style="text-align:center">
                            <img src="https://raw.githubusercontent.com/rhatiro/previsao-renda/main/ebac-course-utils/media/logo/newebac_logo_black_half.png"  width=50%>
                        </div>

                        # **Curso: Ciência de Dados**
                        ### **Streamlit VI e Pycaret**

                        **Por:** [Aline Castro](https://www.linkedin.com/in/alinecastrosantos/%29)<br>
                        **Data:** Janeiro de 2024.<br>

                        ---
                        ''', unsafe_allow_html=True)

    with st.sidebar.expander(label="Bibliotecas utilizadas", expanded=False):
        st.code('''
            Streamlit
            Io
            Numpy 
            Pandas
            Matplotlib.pyplot
            Seaborn
            Pycaret
            SciPy
            Scikit-learn
                
            ''', language='python')



    # Visualização dos dados no corpo principal
    st.markdown('''
                <div style="text-align:center">
                    <img src="https://raw.githubusercontent.com/Aline-Castro/RFV/main/ebac_logo-data_science.png" alt="ebac_logo-data_science" width="100%">
                </div>

                ---

                ### **Módulo 38** | Streamlit VI e Pycaret
                ####  Tarefa II: Construção de um Modelo de Credit Scoring para Cartão de Crédito
                

                **Aluna:** [Aline Castro](https://www.linkedin.com/in/alinecastrosantos/%29)<br>
                **Data:** Janeiro de 2024.

                ---
                ''', unsafe_allow_html=True)

    st.markdown('''
                <a name="intro"></a> 

                Este projeto tem como objetivo desenvolver um modelo de Credit Scoring para avaliação de risco de crédito em solicitações de cartão de crédito.
                Utilizando dados amostrais provenientes de 15 safras, com informações referentes a 12 meses de desempenho, a abordagem adotada visa fornecer uma
                análise preditiva robusta.

                ''', unsafe_allow_html=True)    
    # Carregando a base de dados pela barra lateral
    uploaded_file = st.sidebar.file_uploader("Escolha um arquivo .ftr", type="ftr")

    # Carregando os dados se o arquivo estiver disponível
    st.header('Base de Dados')
    if uploaded_file is not None:
        df = pd.read_feather(uploaded_file)
        st.write(df.head())


##################################### ANÁLISE EXPLORATÓRIA DOS DADOS ###############################################################
        st.header('Análise Exploratória de Dados (EDA)')   
        # Contagem de valores
        st.write('**Contagem de valores únicos**')
        value_counts = df.mau.value_counts(normalize=True)

        # Apresente a contagem de valores no Streamlit
        st.dataframe(value_counts)        
        st.write('**Tipos dos dados**')
        st.write(df.dtypes)
        # Forçando a variável 'qtd_filhos' como numérica
        df.qtd_filhos = df.qtd_filhos.astype(float)
        st.markdown("<h6 style='text-align: left; color: gray;'> qtd de filhos deve ser numérica</h6>", unsafe_allow_html=True)     
      
        # Apresentando a matriz de correlação
        st.write('**Matriz de correlação**')
        corr_matrix = df.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr_matrix, ax=ax)
        st.pyplot(fig)

        # Convertendo o dataframe em uma tabela HTML
        html_corr_matrix = corr_matrix.to_html(index=False)
        # Adicionando tags HTML para centralizar a tabela
        html_centered = f"<center>{html_corr_matrix}</center>"
        # Método st.markdown para exibir a tabela centralizada
        st.markdown(html_centered, unsafe_allow_html=True)

        st.markdown("""
## 

1. **"qtd_filhos" e "qt_pessoas_residencia"**: Essas duas variáveis têm um coeficiente de correlação de 0.890352, indicando uma forte correlação positiva. Isso sugere que à medida que a quantidade de filhos aumenta, a quantidade de pessoas na residência também tende a aumentar.

2. **"idade" e "qtd_filhos"**: Essas duas variáveis têm um coeficiente de correlação de -0.370234, indicando uma correlação negativa moderada. Isso sugere que à medida que a idade aumenta, a quantidade de filhos tende a diminuir.

3. **"tempo_emprego" e "renda"**: Essas duas variáveis têm um coeficiente de correlação de 0.496360, indicando uma correlação positiva moderada. Isso sugere que à medida que o tempo de emprego aumenta, a renda também tende a aumentar.


""")
        
############################ Preparação dos Dados ##########################################################################################################        
        st.header('Preparação dos Dados')
        # Amostragem dos dados
        dataset = df.sample(50000)
        # Removendo colunas desnecessárias
        dataset.drop(['data_ref', 'index'], axis=1, inplace=True)

        # Dividindo os dados em treino e teste
        data = dataset.sample(frac=0.95, random_state=786)
        data_unseen = dataset.drop(data.index)
        data.reset_index(inplace=True, drop=True)
        data_unseen.reset_index(inplace=True, drop=True)

        st.markdown(f"**Conjunto de dados para modelagem (treino e teste):** `{df.shape}`")
        st.markdown(f"**Conjunto de dados não usados no treino/teste, apenas como validação:** `{data_unseen.shape}`")

########################################## MODELAGEM DOS DADOS ######################################################
        st.header('Modelagem dos dados')

        # Configuração do experimento
        exp_clf = setup(data=data, target='mau', session_id=123)

        # Configurando o experimento
        st.write('**Configuração do experimento**')
        exp_aula4 = setup(data=data, target='mau', experiment_name='credit_1',
                          normalize=True, normalize_method='zscore',
                          transformation=True, transformation_method='quantile',
                          fix_imbalance=True)

        # Apresente o resultado no Streamlit
        st.write(exp_aula4)

        # Lista de modelos disponíveis
        models()
        st.write('**Lista de modelos disponíveis**')        
        st.write(models())

        # Treinamento do modelo LightGBM
        lightgbm = create_model('lightgbm')
        tuned_lightgbm = tune_model(lightgbm)
        final_lightgbm = finalize_model(tuned_lightgbm)

        # Avaliação do modelo
        evaluate_model(final_lightgbm)

########################################### AVALIAÇÃO DO MODELO #################################
        st.header("Avaliação do Modelo")
        # Gráfico de Curva ROC
        st.write("**Curva ROC**")
        roc_plot = plot_model(final_lightgbm, plot='auc', save=True)
        st.image(roc_plot)

        st.markdown(""" 
Resultados do gráfico ROC para o LGBMClassifier:

1. **Classe Falsa (linha verde sólida) e Classe Verdadeira (linha azul sólida)**: Ambas têm uma AUC de 0,79. A AUC, ou Área Sob a Curva, é uma métrica de desempenho para classificadores binários. Quanto mais próximo de 1, melhor é o modelo em distinguir entre as duas classes. Um valor de 0,79 indica um bom desempenho do classificador.

2. **Média Micro (linha verde tracejada)**: A média micro leva em conta a frequência de cada classe na computação da média. A AUC de 0,96 para a média micro indica que o classificador tem um excelente desempenho ao considerar a frequência das classes.

3. **Média Macro (linha preta pontilhada)**: A média macro calcula a métrica para cada classe e depois tira a média. A AUC de 0,79 para a média macro indica que o classificador tem um bom desempenho ao considerar todas as classes igualmente.

""")

        # Matriz de Confusão
        st.write("**Matriz de Confusão**")
        confusion_matrix_plot = plot_model(final_lightgbm, plot='confusion_matrix', save=True)
        st.image(confusion_matrix_plot)

        st.markdown(""" 
Detalhes da matriz de confusão para um LGBMClassifier:

- **Verdadeiros Negativos (13087)**: Número de observações negativas corretamente classificadas como negativas pelo modelo.
- **Falsos Positivos (65)**: Número de observações negativas incorretamente classificadas como positivas.
- **Falsos Negativos (1037)**: Nnúmero de observações positivas incorretamente classificadas como negativas.
- **Verdadeiros Positivos (61)**: Número de observações positivas corretamente classificadas como positivas.

""")

        # Gráfico de Precision-Recall
        st.write("**Precision-Recall Curve**")
        pr_plot = plot_model(final_lightgbm, plot='pr', save=True)
        st.image(pr_plot)

        st.markdown(""" 
        A Curva de Precisão-Recall para o LGBMClassifier mostra um equilíbrio entre precisão e revocação. 
        A precisão média é de 0,26, o que indica que, em média, 26% das previsões positivas do modelo são realmente positivas. """)


        # Gráfico de Feature Importance
        st.write("**Importância das Features**")
        feature_importance_plot = plot_model(final_lightgbm, plot='feature', save=True)
        st.image(feature_importance_plot)

        st.markdown(""" 
O gráfico de Importância de Variáveis mostra a importância relativa de cada característica na previsão do modelo, que no caso as mais importantes são:

**Renda**: Esta é a característica mais importante para o modelo, indicando que a renda tem o maior impacto nas previsões do modelo. Isso pode sugerir que a renda é um forte indicador do resultado que o modelo está tentando prever.

**Tempo de Emprego**: Esta é a segunda característica mais importante. Isso sugere que a duração do emprego de uma pessoa também é um fator significativo nas previsões do modelo.""")


        # Remover os arquivos temporários após exibição
        os.remove(roc_plot)
        os.remove(confusion_matrix_plot)
        os.remove(pr_plot)
        os.remove(feature_importance_plot)
  
    
    ################################################################
        # Botão para download do modelo
        st.subheader("Download do Modelo")
        download_button = st.button("Clique aqui para baixar o modelo")

        # Salvar o modelo em um arquivo temporário
        temp_model_file = None
        try:
            if download_button:
                # Avaliando na base out of time
                predict_model(tuned_lightgbm)

                # Preparando para salvar o modelo
                final_lgbm = finalize_model(tuned_lightgbm)

                # Salvando o modelo em um arquivo temporário
                temp_model_file = f"temp_model.pkl"
                with open(temp_model_file, 'wb') as model_file:
                    pickle.dump(final_lgbm, model_file)

                st.success("Modelo baixado com sucesso!")
        except Exception as e:
            st.error(f"Erro ao baixar o modelo: {e}")
        finally:
            # Remover o arquivo temporário após exibição
            if temp_model_file and os.path.exists(temp_model_file):
                os.remove(temp_model_file)

# Desativar o aviso sobre o uso global do Pyplot
st.set_option('deprecation.showPyplotGlobalUse', False)
        

if __name__ == '__main__':
    main()
