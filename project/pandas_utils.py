import os
import unicodedata

import pandas as pd


def get_name_from_path(video_path):
    name = video_path.split('/')[-1]
    name = name.split('.')[0]
    return name


def get_result_from_excel(df, name_video):
    result = df[df['Número do Vídeo'].str.strip() == name_video]['Resultado Esperado'].values
    if len(result) == 0:
        print(f"Erro: Nenhum resultado encontrado para o vídeo {name_video}")
        return []  # Retorna uma lista vazia caso não encontre o vídeo

    # Formata o resultado
    formated_result = str(result[0]).split(',')[-1].upper().strip()
    formated_result =  ''.join(
        c if c == 'Ç' else c for c in unicodedata.normalize('NFD', formated_result)
        if unicodedata.category(c) != 'Mn' or c == 'Ç'
    )
    formated_result_arr = formated_result.split('E')
    formated_result_arr = map(str.strip, formated_result_arr)


    return formated_result_arr

def contains_register(name_video):
    if not os.path.exists('./res/result/tabela_resultados.xlsx'):
        return False
    df = pd.read_excel('res/result/tabela_resultados.xlsx')
    return (df['Id'] == name_video).any()
def save_or_update_table(new_data, base_bath='res/result/', file_name='tabela_resultados'):
    # Verificando se o arquivo já existe
    file_path = os.path.join(base_bath, file_name+'.xlsx')
    if os.path.exists(file_path):
        existing_df = pd.read_excel(file_path)  # Substituído para ler arquivo Excel
        updated_df = pd.concat([existing_df, pd.DataFrame(new_data)], ignore_index=True)
    else:
        updated_df = pd.DataFrame(new_data)

    updated_df.to_excel(base_bath, index=False)
    print(updated_df)
    return base_bath



