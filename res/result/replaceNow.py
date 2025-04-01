import pandas as pd

df = pd.read_excel('/home/athos/PycharmProjects/OcrNutritionInfo/res/result/tabela_resultados_1_replaced.xlsx')


def calcular_acertos(resultado_obtido, resultado_esperado, sucedido):
    if not isinstance(resultado_obtido, str):
        return 0
    # Dividir as linhas em listas
    resultado_obtido = resultado_obtido.split(', ')
    resultado_esperado = resultado_esperado.split(' , ')


    if len(resultado_esperado) == 1 and sucedido == True:
        return 1
    elif len(resultado_esperado) > 1 and sucedido == False:
        # Iterar sobre os valores de resultado_obtido
        for valor in resultado_obtido:
            # Verifica se algum valor de resultado_obtido está em resultado_esperado

            if valor in resultado_esperado:
                return 0.5
    return 0
# Função para calcular a similaridade entre as duas colunas e avaliar a pontuação


# Aplicar a função e calcular a taxa de acerto
values = [calcular_acertos(row['Resultado Obtido'], row['Resultado Esperado'], row['Sucedido'] ) for _,row in df.iterrows()]
print(len(values))
print(values)
soma = sum(values)
print(soma)
print((soma / len(values)) * 100 )
