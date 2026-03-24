import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from data_loader import carregar_dados

df = carregar_dados()

# ao inves de usar aeroporto como categorico, criar média de atraso por aeroporto
airport_delay = df.groupby('ORIGIN_AIRPORT')['DEPARTURE_DELAY'].mean()

df['AIRPORT_DELAY_MEAN'] = df['ORIGIN_AIRPORT'].map(airport_delay)

# features
features = [
    'DISTANCE',
    'SCHEDULED_DEPARTURE_HR',
    'AIRPORT_DELAY_MEAN',
    'AIRLINE'
]

df_model = df[features].dropna().copy()

# padronizacao e normalizacao
df_encoded = pd.get_dummies(df_model, columns=['AIRLINE'], drop_first=True)

df_encoded = df_encoded.select_dtypes(include=['number'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded)

# metodo do cotovelo
inertia = []

k_range = range(1, 10)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(k_range, inertia, marker='o')
plt.xlabel('Número de clusters (k)')
plt.ylabel('Inertia')
plt.title('Método do Cotovelo')
plt.show()

# modelo final com k=4
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df_model['cluster'] = kmeans.fit_predict(X_scaled)

# media de atraso por cluster
print("\nMédia de atraso por cluster:")
df_cluster = df_model.merge(
    df[['DEPARTURE_DELAY']],
    left_index=True,
    right_index=True,
    how='left'
)
print(df_cluster.groupby('cluster')['DEPARTURE_DELAY'].mean())

# media das features por cluster
print("\nMédia das features numéricas por cluster:")
print(df_model.groupby('cluster')[[
    'DISTANCE',
    'SCHEDULED_DEPARTURE_HR',
    'AIRPORT_DELAY_MEAN'
]].mean())

# média de atraso por cluster e por companhia aérea (feature categorica)
pivot = df_cluster.pivot_table(
    values='DEPARTURE_DELAY',
    index='cluster',
    columns='AIRLINE',
    aggfunc='mean'
)

plt.figure(figsize=(12,6))
sns.heatmap(pivot, annot=True, fmt=".1f")
plt.title('Média de atraso por cluster e companhia aérea')
plt.xlabel('Airline')
plt.ylabel('Cluster')
plt.show()