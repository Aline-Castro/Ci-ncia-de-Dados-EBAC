import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Configuração do tema do seaborn
sns.set_theme()

def plot_pivot_table(df: pd.DataFrame, value: str,  index: str,  func: str, ylabel: str, xlabel: str,  opcao: str='nenhuma') -> None:
    # Função para criar e plotar uma tabela dinâmica
    if opcao == 'nenhuma':
        pivot = pd.pivot_table(data=df, values=value, index=index, aggfunc=func)
    elif opcao == 'sort':
        pivot = pd.pivot_table(data=df, values=value, index=index, aggfunc=func).sort_values(value)
    elif opcao == 'unstack':
        pivot = pd.pivot_table(data=df, values=value, index=index, aggfunc=func).unstack()

    pivot.plot(figsize=[15, 5])
    plt.ylabel(ylabel, color='purple')  
    plt.xlabel(xlabel, color='purple')  
    plt.title(value + ' por ' + xlabel, color='navy')
    return None

def main(meses_input):
    for mes in meses_input:
        # Carregando o arquivo csv do mês especificado
        sinasc = pd.read_csv(f'https://raw.githubusercontent.com/rhatiro/Curso_EBAC-Profissao_Cientista_de_Dados/main/Mo%CC%81dulo%2014%20-%20Scripting/database/input/SINASC_RO_2019_{mes}.csv')

        max_data = sinasc.DTNASC.max()[:7]
        os.makedirs('./output/figs/'+max_data, exist_ok=True)

        # Plotando gráficos para cada métrica e salvando como arquivos .png
        plot_pivot_table(df=sinasc, value='IDADEMAE', index='DTNASC', func='count', ylabel='Quantidade de nascimentos', xlabel='Data de nascimento')
        plt.savefig('./output/figs/'+max_data+'/Quantidade de nascimentos.png')
        plt.close()

        plot_pivot_table(df=sinasc, value='IDADEMAE', index=['DTNASC', 'SEXO'], func='mean', ylabel='Média da idade das mães', xlabel='Data de nascimento', opcao='unstack')
        plt.savefig('./output/figs/'+max_data+'/Média da idade das mães por sexo.png')
        plt.close()

        plot_pivot_table(df=sinasc, value='PESO', index=['DTNASC', 'SEXO'], func='mean', ylabel='Média do peso dos bebês', xlabel='Data de nascimento', opcao='unstack')
        plt.savefig('./output/figs/'+max_data+'/Média do peso dos bebês por sexo.png')
        plt.close()

        plot_pivot_table(df=sinasc, value='APGAR1', index='ESCMAE', func='median', ylabel='Mediana do APGAR1', xlabel='Escolaridade', opcao='sort')
        plt.savefig('./output/figs/'+max_data+'/Mediana do APGAR1 por escolaridade das mães.png')
        plt.close()

        plot_pivot_table(df=sinasc, value='APGAR1', index='GESTACAO', func='mean', ylabel='Média do APGAR1', xlabel='Gestação', opcao='sort')
        plt.savefig('./output/figs/'+max_data+'/Média do APGAR1 por gestação.png')
        plt.close()

        plot_pivot_table(df=sinasc, value='APGAR5', index='GESTACAO', func='mean', ylabel='Média do APGAR5', xlabel='Gestação', opcao='sort')
        plt.savefig('./output/figs/'+max_data+'/Média do APGAR5 por gestação.png')
        plt.close()
        
        print('Data inicial:', sinasc.DTNASC.min())
        print('Data final:', sinasc.DTNASC.max())
        print('Nome da pasta:', max_data, '\n')

if __name__ == "__main__":
    main(sys.argv[1:])
