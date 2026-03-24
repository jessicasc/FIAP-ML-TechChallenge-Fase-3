import pandas as pd
import matplotlib.pyplot as plt

# seleciono as colunas que vou usar nos modelos para feature e target
# converto as colunas ORIGIN_AIRPORT e DESTINATION_AIRPORT em string (pois elas possuem mais de um tipo de dados)
df = pd.read_csv(
    '../data/flights.csv',
    usecols=[
        'MONTH','DAY','DAY_OF_WEEK','AIRLINE',
        'ORIGIN_AIRPORT','DESTINATION_AIRPORT',
        'SCHEDULED_DEPARTURE','SCHEDULED_ARRIVAL',
        'SCHEDULED_TIME','DISTANCE','DEPARTURE_DELAY', 
        'CANCELLED', 'DIVERTED'
    ],
    dtype={
        'ORIGIN_AIRPORT': 'str',
        'DESTINATION_AIRPORT': 'str'
    },
    low_memory=False
)

# removo os voos cancelados ou desviados
df = df[(df['CANCELLED'] == 0) & (df['DIVERTED'] == 0)]

# crio a coluna target que será usada no modelo
# 0 voo no horario ou adiantado, 1 voo atrasado
df['TARGET_DELAY'] = (df['DEPARTURE_DELAY'] > 0).astype(int)

# tratar horas
df['SCHEDULED_DEPARTURE_HR'] = df['SCHEDULED_DEPARTURE'] // 100
df['SCHEDULED_ARRIVAL_HR'] = df['SCHEDULED_ARRIVAL'] // 100
df['SCHEDULED_TIME_HR'] = df['SCHEDULED_TIME'] // 100

# quantidade de linhas e tipos de dados
print("Shape do dataset:", df.shape)

print("\nTipos de dados:\n", df.dtypes)

# sem valores nulos
print("\nValores nulos por coluna:")
print(df.isnull().sum())

# poucos registros duplicados
print("\nLinhas duplicadas:", df.duplicated().sum())

# correlação das colunas numéricas com target (atraso)
numeric_cols = [
    'MONTH', 'DAY', 'DAY_OF_WEEK',
    'SCHEDULED_DEPARTURE_HR', 'SCHEDULED_ARRIVAL_HR',
    'SCHEDULED_TIME_HR', 'DISTANCE', 'TARGET_DELAY'
]

print("\nCorrelação com target:")
corr = df[numeric_cols].corr()

print(corr['TARGET_DELAY'].sort_values(ascending=False))

# distribuição do atraso
plt.figure()
df['TARGET_DELAY'].value_counts().plot(kind='bar')
plt.title("Distribuição do Target (Atraso)")
plt.xlabel("Classe (0 = Não, 1 = Sim)")
plt.ylabel("Quantidade")
plt.xticks(rotation=0)
plt.grid(True)
plt.ticklabel_format(style='plain', axis='y')
plt.show()

# atraso por dia
plt.figure()
df.groupby('DAY')['TARGET_DELAY'].mean().plot(kind='bar')
plt.title("Atraso por dia")
plt.xlabel("Dia")
plt.ylabel("Taxa de atraso")
plt.xticks(rotation=0)
plt.grid(True)
plt.show()

# atraso por dia da semana
plt.figure()
df.groupby('DAY_OF_WEEK')['TARGET_DELAY'].mean().plot(kind='bar')
plt.title("Atraso por dia da semana")
plt.xlabel("Dia da semana")
plt.ylabel("Taxa de atraso")
plt.xticks(rotation=0)
plt.grid(True)
plt.show()

# atraso por mês
plt.figure()
df.groupby('MONTH')['TARGET_DELAY'].mean().plot(kind='bar')
plt.title("Atraso por mês")
plt.xlabel("Mês")
plt.ylabel("Taxa de atraso")
plt.xticks(rotation=0)
plt.grid(True)
plt.show()

# atraso por hora de partida programada
plt.figure()
df.groupby('SCHEDULED_DEPARTURE_HR')['TARGET_DELAY'].mean().plot(kind='bar')
plt.title("Atraso por hora de partida programada")
plt.xlabel("Partida programada")
plt.ylabel("Taxa de atraso")
plt.xticks(rotation=0)
plt.grid(True)
plt.show()

# atraso por companhia aerea
plt.figure()
df.groupby('AIRLINE')['TARGET_DELAY'].mean().plot(kind='bar')
plt.title("Atraso por companhia aerea")
plt.xlabel("Companhia")
plt.ylabel("Taxa de atraso")
plt.xticks(rotation=0)
plt.grid(True)
plt.show()

# atraso por aeroporto (partindo dos 10 com mais atrasos)
top_airport = df['ORIGIN_AIRPORT'].value_counts().head(10).index

plt.figure()
df[df['ORIGIN_AIRPORT'].isin(top_airport)] \
    .groupby('ORIGIN_AIRPORT')['TARGET_DELAY'].mean() \
    .sort_values() \
    .plot(kind='bar')

plt.title("Atraso por aeroporto (top 10)")
plt.xlabel("Aeroportos")
plt.ylabel("Taxa de atraso")
plt.xticks(rotation=0)
plt.grid(True)
plt.show()

# atraso de acordo com a distancia do voo
df['DISTANCE_BIN'] = pd.qcut(df['DISTANCE'], q=5)

plt.figure()
df.groupby('DISTANCE_BIN')['TARGET_DELAY'] \
    .mean() \
    .plot(kind='bar')

plt.title("Atraso por faixa de distância")
plt.xlabel("Faixa de distância")
plt.ylabel("Taxa de atraso")
plt.xticks(rotation=0)
plt.grid(True)
plt.show()