# IMPORTANDO AS BIBLIOTECAS NECESSÁRIAS
import streamlit             as st
import io
import numpy                 as np
import pandas                as pd
import matplotlib.pyplot     as plt
import seaborn               as sns
from gower                   import gower_matrix
from scipy.spatial.distance  import squareform
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


@st.cache_data(show_spinner=False)
def calcularGowerMatrix(data_x, cat_features):
    return gower_matrix(data_x=data_x, cat_features=cat_features)

@st.cache_data(show_spinner=False)
# Definir a função para criar um dendrograma
def dn(color_threshold: float, num_groups: int, Z: list) -> None:
    """
    Cria e exibe um dendrograma.

    Parameters:
        color_threshold (float): Valor de threshold de cor para a coloração do dendrograma.
        num_groups (int): Número de grupos para o título do dendrograma.
        Z (list): Matriz de ligação Z.

    Returns:
        None
    """
    plt.figure(figsize=(24, 6))
    plt.ylabel(ylabel='Distância')
    
    # Adicionar o número de grupos como título
    plt.title(f'Dendrograma Hierárquico - {num_groups} Grupos')

    # Criar o dendrograma com base na matriz de ligação Z
    dn = dendrogram(Z=Z, 
                    p=6, 
                    truncate_mode='level', 
                    color_threshold=color_threshold, 
                    show_leaf_counts=True, 
                    leaf_font_size=8, 
                    leaf_rotation=45, 
                    show_contracted=True)
    plt.yticks(np.linspace(0, .6, num=31))
    plt.xticks([])

    # Exibir o dendrograma criado
    st.pyplot(plt)

    # Imprimir o número de elementos em cada parte do dendrograma
    for i in dn.keys():
        st.text(f'dendrogram.{i}: {len(dn[i])}')


# Função principal da aplicação
def main():
    # Configuração inicial da página da aplicação
    st.set_page_config(
        page_title="EBAC | Módulo 31 | Projeto de Agrupamento hierárquico",
        page_icon='https://raw.githubusercontent.com/Aline-Castro/RFV/main/favicon.ico', 
        layout="wide",
        initial_sidebar_state="expanded",
    )


    st.sidebar.markdown('''
                        <div style="text-align:center">
                            <img src="https://raw.githubusercontent.com/rhatiro/previsao-renda/main/ebac-course-utils/media/logo/newebac_logo_black_half.png"  width=50%>
                        </div>

                        # **Curso: Ciência de Dados**
                        ### **Projeto de Agrupamento Hierárquico**

                        **Por:** [Aline Castro](https://www.linkedin.com/in/alinecastrosantos/)<br>
                        **Data:** Dezembro de 2023.<br>

                        ---
                        ''', unsafe_allow_html=True)
    

    with st.sidebar.expander(label="Índice", expanded=False):
        st.markdown('''
                    - [1. Visualização dos Dados](#visualizacao)
                        - [1.1 Primeiras linhas dos dados do arquivo carregado](#inicio)
                        - [1.2 Quantidade de sessões que resultaram em receita](#sessao_receita)
                        - [1.3 Representação gráfica da contagem de 'Revenue'](#plot_cont)
                    - [2. Análise Descritiva](#descritiva)
                        - [2.1 Informações sobre a estrutura do DataFrame](#info)
                        - [2.2 Visualização gráfica da distribuição das variáveis](#distrplot)
                        - [2.3 Representação gráfica da correlação entre as variáveis](#corrplot)
                    - [3. Variáveis de agrupamento](#agr_var)
                        - [3.1 Seleção das variáveis](#var_sel)
                            - [3.1.1 Variáveis que descrevam o padrão de navegação na sessão](#var_sel_pad)
                            - [3.1.2 Variáveis que indicam a característica da data](#var_sel_carac)
                        - [3.2 Novo DataFrame com as variáveis tratadas](#var_tratada)
                    - [4. Número de grupos](#num_grup)
                        - [4.1 Agrupamento hierárquico e dendrograma para visualizar os resultados](#dengograma)
                        - [4.2 Agrupamento hierárquico com 3 grupos](#sagrup3)
                        - [4.3 Agrupamento hierárquico com 4 grupos](#agrup4)
                    - [5. Avaliação dos grupos](#avaliacao)
                    - [6. Avaliação de resultados](#resultado)
                    
                    ''', unsafe_allow_html=True)


    with st.sidebar.expander(label="Bibliotecas utilizadas", expanded=False):
        st.code('''
                Streamlit
                Io
                Numpy 
                Pandas
                Matplotlib.pyplot
                Seaborn
                Gower
                SciPy
                Scikit-learn
                
                ''', language='python')
        

    st.sidebar.markdown('''
                        ---
                        *Baseado no [Exercício 2](https://github.com/Aline-Castro/Ciencia-de-Dados/blob/main/Profiss%C3%A3o%3A%20Cientista%20de%20Dados/M%C3%B3dulo%2030%20-%20Hier%C3%A1rquicos%20Aglomerativos%20/Exerc%C3%ADcio%202/mod30_tarefa02.ipynb) do Módulo 30.*
                        ''')


    st.markdown('''
                <div style="text-align:center">
                    <img src="https://raw.githubusercontent.com/Aline-Castro/RFV/main/ebac_logo-data_science.png" alt="ebac_logo-data_science" width="100%">
                </div>

                ---

                <!-- # **Profissão: Cientista de Dados** -->
                ### **Módulo 31** | Streamlit V | Exercício 2

                **Aluna:** [Aline Castro](https://www.linkedin.com/in/alinecastrosantos/)<br>
                **Data:** Dezembro de 2023.

                ---
                ''', unsafe_allow_html=True)


    st.markdown('''
                <a name="intro"></a> 

                # Agrupamento hierárquico

                Neste projeto foi utilizada a base [online shoppers purchase intention](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset) de Sakar, C.O., Polat, S.O., Katircioglu, M. et al. Neural Comput & Applic (2018). [Web Link](https://doi.org/10.1007/s00521-018-3523-0).

                A base trata de registros de 12.330 sessões de acesso a páginas, cada sessão sendo de um único usuário em um período de 12 meses, para posteriormente relacionar o design da página e o perfil do cliente.
                
                ***"Será que clientes com comportamento de navegação diferentes possuem propensão a compra diferente?"***

                O objetivo é agrupar as sessões de acesso ao portal considerando o comportamento de acesso e informações da data, como a proximidade a uma data especial, fim de semana e o mês.

                |Variável                |Descrição                                                                                                                      |Atributo   | 
                | :--------------------- |:----------------------------------------------------------------------------------------------------------------------------  | --------: | 
                |Administrative          | Quantidade de acessos em páginas administrativas                                                                              |Numérico   | 
                |Administrative_Duration | Tempo de acesso em páginas administrativas                                                                                    |Numérico   | 
                |Informational           | Quantidade de acessos em páginas informativas                                                                                 |Numérico   | 
                |Informational_Duration  | Tempo de acesso em páginas informativas                                                                                       |Numérico   | 
                |ProductRelated          | Quantidade de acessos em páginas de produtos                                                                                  |Numérico   | 
                |ProductRelated_Duration | Tempo de acesso em páginas de produtos                                                                                        |Numérico   | 
                |BounceRates             | *Percentual de visitantes que entram no site e saem sem acionar outros *requests* durante a sessão                            |Numérico   | 
                |ExitRates               | * Soma de vezes que a página é visualizada por último em uma sessão dividido pelo total de visualizações                      |Numérico   | 
                |PageValues              | * Representa o valor médio de uma página da Web que um usuário visitou antes de concluir uma transação de comércio eletrônico |Numérico   | 
                |SpecialDay              | Indica a proximidade a uma data festiva (dia das mães etc)                                                                    |Numérico   | 
                |Month                   | Mês                                                                                                                           |Categórico | 
                |OperatingSystems        | Sistema operacional do visitante                                                                                              |Categórico | 
                |Browser                 | Browser do visitante                                                                                                          |Categórico | 
                |Region                  | Região                                                                                                                        |Categórico | 
                |TrafficType             | Tipo de tráfego                                                                                                               |Categórico | 
                |VisitorType             | Tipo de visitante: novo ou recorrente                                                                                         |Categórico | 
                |Weekend                 | Indica final de semana                                                                                                        |Categórico | 
                |Revenue                 | Indica se houve compra ou não                                                                                                 |Categórico |

                *Variáveis calculadas pelo Google Analytics*

                ''', unsafe_allow_html=True)


    st.markdown(''' 
                ## 1. Visualização dos Dados
                <a name="visualizacao"></a> 
                ''', unsafe_allow_html=True)
    

    st.markdown(''' 
                ### 1.1. Primeiras linhas dos dados do arquivo carregado:
                <a name="inicio"></a> 
                ''', unsafe_allow_html=True)
    
    # Lendo o arquivo CSV 'online_shoppers_intention.csv' e armazenando os dados em um DataFrame chamado df
    df = pd.read_csv('https://raw.githubusercontent.com/Aline-Castro/Agrupamento-Hierarquico/main/online_shoppers_intention.csv')

    # Exibindo o DataFrame df, mostrando os dados carregados do arquivo CSV
    st.dataframe(df.head())

    st.markdown(''' 
                ### 1.2. Quantidade de sessões que resultaram em receita (compra) e as que não resultaram:
                <a name="sessao_receita"></a> 
                ''', unsafe_allow_html=True)
    st.dataframe(df.Revenue.value_counts(dropna=False))
    
    # O argumento dropna=False garante que também serão contadas as entradas que possuem valores NaN.
    st.markdown(''' 
                ### 1.3. Representação gráfica da contagem de 'Revenue' 
                <a name="plot_cont"></a>
                ''', unsafe_allow_html=True)
    # Criando um gráfico de contagem (count plot) para a coluna 'Revenue' usando seaborn
    sns.countplot(x='Revenue', data=df)

    # Exibindo o gráfico
    st.pyplot(plt)

    st.markdown('''
    <p style="font-size:12px">
    False: Indica o número de sessões que não resultaram em uma compra. No caso, houve 10.422 sessões que não geraram receita.
    <br>
    True: Indica o número de sessões que resultaram em uma compra. No caso, houve 1.908 sessões que geraram receita.
    </p>
    ''', unsafe_allow_html=True)

    st.markdown(''' 
                ## 2. Análise Descritiva
                <a name="descritiva"></a>
                ''', unsafe_allow_html=True)
    
    st.write(df.describe())

    st.markdown(''' 
                ### 2.1. Informações sobre a estrutura do DataFrame
                <a name="info"></a>
                ''', unsafe_allow_html=True)
    # Imprimir informações sobre a estrutura do DataFrame
    st.info(f''' 
            Quantidade de linhas: {df.shape[0]}

            Quantidade de colunas: {df.shape[1]}

            Quantidade de valores missing: {df.isna().sum().sum()} 
            ''')

    st.markdown(''' 
                ### 2.2. Visualização gráfica da distribuição das variáveis
                <a name="distrplot"></a>
                ''', unsafe_allow_html=True)   


    # Convertendo colunas booleanas para inteiros
    for col in df.select_dtypes(include=['bool']).columns:
        df[col] = df[col].astype(int)

    # Verificando a distribuição das variáveis
    fig, axs = plt.subplots(len(df.columns)//2, 2, figsize=(15, len(df.columns)*2))

    for i, column in enumerate(df.columns):
        axs[i//2, i%2].hist(df[column], bins=30)
        axs[i//2, i%2].set_title(column, color = 'navy')

    plt.tight_layout()
    st.pyplot(plt)

    # Limpar a figura
    plt.clf()

    st.markdown(''' 
                ### 2.3. Representação gráfica da correlação entre as variáveis
                <a name="corrplot"></a>
                ''', unsafe_allow_html=True)

    # Criar um mapa de calor (heatmap) para visualizar a correlação entre as colunas do DataFrame
    plt.figure(figsize=(10, 10))  # Ajuste os valores conforme necessário
    sns.heatmap(df.corr(numeric_only=True), cmap='viridis')

    # Exibir o mapa de calor
    st.pyplot(plt)

    st.markdown(''' 
                ## 3. Variáveis de agrupamento
                <a name="agr_var"></a>
                ''', unsafe_allow_html=True)
    st.markdown(''' 
                ### 3.1. Seleção das variáveis
                <a name="var_sel"></a>
                ''', unsafe_allow_html=True)

    st.markdown(''' 
                #### 3.1.1. Variáveis que descrevam o padrão de navegação na sessão:
                <a name="var_sel_pad"></a>
                ''', unsafe_allow_html=True)
    
    variaveis = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
             'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues','SpecialDay', 'Month', 'Weekend']   
    # Lista de variáveis que descrevem o padrão de navegação na sessão

    variaveis_qtd = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
             'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues']
    
    st.markdown('''
    <p style="font-size:16px">
    Para o agrupamento, as seguintes variáveis quantitativas, que descrevem o padrão de navegação na sessão serão consideradas:
    <ul>
        <li><b>Administrative</b>: Quantidade de acessos em páginas administrativas.</li>
        <li><b>Administrative_Duration</b>: Tempo de acesso em páginas administrativas.</li>
        <li><b>Informational</b>: Quantidade de acessos em páginas informativas.</li>
        <li><b>Informational_Duration</b>: Tempo de acesso em páginas informativas.</li>
        <li><b>ProductRelated</b>: Quantidade de acessos em páginas de produtos.</li>
        <li><b>ProductRelated_Duration</b>: Tempo de acesso em páginas de produtos.</li>
        <li><b>BounceRates</b>: Percentual de visitantes que entram no site e saem sem acionar outros requests durante a sessão.</li>
        <li><b>ExitRates</b>: Soma de vezes que a página é visualizada por último em uma sessão dividido pelo total de visualizações.</li>
        <li><b>PageValues</b>: Representa o valor médio de uma página da Web que um usuário visitou antes de concluir uma transação de comércio eletrônico.</li>
    </ul>
    </p>
    ''', unsafe_allow_html=True)

    st.markdown(''' 
                #### 3.1.2. Variáveis que indicam a característica da data:
                <a name="var_sel_carac"></a>
                ''', unsafe_allow_html=True)

    variaveis_cat = ['SpecialDay', 'Month', 'Weekend']
    
    st.markdown('''
    <p style="font-size:16px">
    Variáveis quantitativas, que indicam a característica da data,são:
    <ul>
        <li><b>SpecialDay:</b> Indica a proximidade a uma data festiva (dia das mães etc).</li>
        <li><b>Month:</b> Mês.</li>
        <li><b>Weekend:</b> Indica final de semana.</li>
    </ul>
    </p>
    ''', unsafe_allow_html=True)    

    st.markdown(''' 
                ### 3.2. Novo DataFrame com as variáveis tratadas
                <a name="var_tratada"></a>
                ''', unsafe_allow_html=True)
    
    # Criando um novo DataFrame que contém as variáveis quantitativas padronizadas (média 0 e desvio padrão 1)
    # e as variáveis categóricas originais do DataFrame 'df'.
    df_padrao = pd.DataFrame(StandardScaler().fit_transform(df[variaveis_qtd]), columns = df[variaveis_qtd].columns)
    df_padrao[variaveis_cat] = df[variaveis_cat]

    # Cria um novo DataFrame 'df_tratado' que contém as variáveis selecionadas do DataFrame 'df_padrao'
    # com todas as variáveis categóricas convertidas em variáveis dummy e todas as linhas contendo valores NaN removidas.

    df_tratado = pd.get_dummies(df_padrao[variaveis].dropna(), columns = variaveis_cat)
    st.dataframe(df_tratado.head())

    st.markdown(''' 
                ##  4. Número de grupos
                <a name="num_grup"></a>
                ''', unsafe_allow_html=True)
    
    st.markdown(''' 
                ###  4.1. Dendrograma para visualizar os resultados do agrupamento hierárquico:
                <a name="dengograma"></a>
                ''', unsafe_allow_html=True)    

    # Realizando o agrupamento hierárquico
    linked = linkage(df_tratado, 'ward')
    # Gerando o dendrograma
    plt.figure(figsize=(10, 7))
    dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
    st.pyplot(plt)
   
    st.markdown(''' 
                ###  4.2. Agrupamento hierárquico com 3 grupos:
                <a name="sagrup3"></a>
                ''', unsafe_allow_html=True)
    cluster3 = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
    st.write(cluster3.fit_predict(df_tratado))

    st.markdown(''' 
                ###  4.3. Agrupamento hierárquico com 4 grupos:
                <a name="agrup4"></a>
                ''', unsafe_allow_html=True)
    cluster4 = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
    st.write(cluster4.fit_predict(df_tratado))


    st.markdown(''' 
                ##  5. Avaliação dos grupos
                <a name="avaliacao"></a>
                ''', unsafe_allow_html=True)

    # Padronização das variáveis quantitativas
    scaler = StandardScaler()
    df[variaveis_qtd] = scaler.fit_transform(df[variaveis_qtd])

    # Tratamento de valores faltantes
    df = df.dropna()

    # Transformação de variáveis qualitativas em variáveis dummy
    # Transformação de todas as variáveis categóricas em variáveis dummy
    df = pd.get_dummies(df)

    # Construção dos agrupamentos
    for n_clusters in [3, 4]:
        clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        labels = clusterer.fit_predict(df)

        # Avaliação dos agrupamentos
        silhouette_avg = silhouette_score(df, labels)
        st.write("Para n_clusters =", n_clusters, "o score médio do coeficiente de silhueta é :", silhouette_avg)
        st.write('\n')

        # Análise descritiva dos agrupamentos
        df['Cluster'] = labels
        st.write(df.groupby('Cluster').mean())

    st.markdown('''
    <p style="font-size:16px">
    O coeficiente de silhueta médio para 3 grupos é 0.26 e para 4 grupos é 0.27.<br>
    O coeficiente de silhueta varia de -1 a 1, e valores mais altos indicam que os pontos estão mais próximos do seu próprio cluster do que dos outros clusters, sugerindo um bom agrupamento.<br>
    Com base nesses resultados, considerando que o coeficiente de silhueta seja mais importante, o agrupamento de 4 clusters seria a melhor opção, pois tem um coeficiente de silhueta médio mais alto.

    </p>
    ''', unsafe_allow_html=True)

    st.markdown(''' 
                ##  6. Avaliação de resultados
                <a name="resultado"></a>
                ''', unsafe_allow_html=True)
    
    # Calculando a média de 'BounceRates' e 'Revenue' para cada grupo
    st.dataframe(df.groupby('Cluster')[['BounceRates', 'Revenue']].mean())


    st.markdown('''
    <p style="font-size:16px">
    Os resultados mostram a média de BounceRates e Revenue para cada grupo (Cluster 0, 1, 2 e 3).
    <ul>
        <li> O Cluster 0 tem uma taxa média de rejeição (BounceRates) de 0.0266 e uma proporção média de sessões que resultaram em uma compra (Revenue) de 0.1534.</li>
        <li> O Cluster 1 tem uma taxa média de rejeição de 0.1143 e uma proporção média de sessões que resultaram em uma compra de 0.2257.
        <li> O Cluster 2 tem uma taxa média de rejeição de -0.2742 e uma proporção média de sessões que resultaram em uma compra de 0.1540.
        <li> O Cluster 3 tem uma taxa média de rejeição de 0.1703 e uma proporção média de sessões que resultaram em uma compra de 0.1527.
    </ul>     
        Com base nesses resultados, o grupo com clientes mais propensos à compra seria o Cluster 1, pois tem o maior valor médio para Revenue (0.225).<br>
        Isso significa que, em média, 22.57% das sessões neste grupo resultaram em uma compra.

    </p>
    ''', unsafe_allow_html=True)

    
    
    




    

    






















if __name__ == '__main__':
    main()               