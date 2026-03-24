import pandas as pd

def carregar_dados(caminho='../data/flights.csv'):

    df = pd.read_csv(
        caminho,
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

    # Remover voos cancelados ou desviados
    df = df[(df['CANCELLED'] == 0) & (df['DIVERTED'] == 0)]

    # Criar target
    df['TARGET_DELAY'] = (df['DEPARTURE_DELAY'] > 0).astype(int)

    # Tratar horas
    df['SCHEDULED_DEPARTURE_HR'] = df['SCHEDULED_DEPARTURE'] // 100
    df['SCHEDULED_ARRIVAL_HR'] = df['SCHEDULED_ARRIVAL'] // 100
    df['SCHEDULED_TIME_HR'] = df['SCHEDULED_TIME'] // 100

    # tratar duplicatas
    df = df.drop_duplicates()

    return df