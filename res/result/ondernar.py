import pandas as pd
import pandas as pd


def ordenar_videos(arquivo_entrada, arquivo_saida):
    """
    Ordena a coluna 'Vídeos' numericamente em ordem crescente.
    Mantém todas as linhas (inclusive duplicatas).

    Parâmetros:
        arquivo_entrada (str): Caminho do arquivo Excel original.
        arquivo_saida (str): Caminho para salvar o Excel ordenado.
    """
    df = pd.read_excel(arquivo_entrada)
    df['Número'] = df['Id'].str.extract('(\d+)').astype(int)

    # 3. Ordenar pelo número em ordem crescente
    df_ordenado = df.sort_values("Número")

    # 4. Remover a coluna auxiliar (se não quiser no arquivo final)
    df_final = df_ordenado.drop(columns=["Número"])

    df_ordenado.to_excel(arquivo_saida, index=False)
    print(f"Arquivo ordenado salvo em: {arquivo_saida}")


def ordenar_e_remover_duplicatas(arquivo_entrada, arquivo_saida):
    df = pd.read_excel(arquivo_entrada)

    # Extrai números, ordena e remove duplicatas
    df['Número'] = df['Id'].str.extract('(\d+)').astype(int)
    df_final = (
        df.sort_values('Número')
        .drop_duplicates(subset='Número')
        .drop(columns='Número')
    )

    df_final.to_excel(arquivo_saida, index=False)
    print(f"Arquivo ordenado e sem duplicatas salvo em: {arquivo_saida}")




def remover_duplicatas(arquivo_entrada, arquivo_saida, coluna='Id'):
    """
    Remove linhas duplicadas da coluna especificada (padrão: 'Vídeos'),
    mantendo a primeira ocorrência e a ordem original do arquivo.

    Parâmetros:
        arquivo_entrada (str): Caminho do arquivo Excel original.
        arquivo_saida (str): Caminho para salvar o arquivo sem duplicatas.
        coluna (str): Nome da coluna a ser verificada (opcional, padrão é 'Vídeos').
    """
    # Lê o arquivo Excel
    df = pd.read_excel(arquivo_entrada)

    # Remove duplicatas (mantém a primeira ocorrência)
    df_sem_duplicatas = df.drop_duplicates(subset=coluna)

    # Salva o resultado
    df_sem_duplicatas.to_excel(arquivo_saida, index=False)
    print(f"✅ Arquivo sem duplicatas salvo em: '{arquivo_saida}'")



ordenar_videos("Teste 01.xlsx", "Teste 01_ord.xlsx")

