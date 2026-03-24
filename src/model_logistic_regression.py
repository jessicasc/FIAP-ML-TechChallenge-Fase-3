from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from data_loader import carregar_dados

df = carregar_dados()

# features e target (atraso)
features = [
    'MONTH',
    'DAY',
    'DAY_OF_WEEK',
    'AIRLINE',
    'ORIGIN_AIRPORT',
    'DESTINATION_AIRPORT',
    'SCHEDULED_DEPARTURE_HR',
    'SCHEDULED_ARRIVAL_HR',
    'SCHEDULED_TIME_HR',
    'DISTANCE'
]

X = df[features]
y = df['TARGET_DELAY']

# separar tipo de coluna
categorical_features = [
    'AIRLINE',
    'ORIGIN_AIRPORT',
    'DESTINATION_AIRPORT'
]

numeric_features = [
    'MONTH',
    'DAY',
    'DAY_OF_WEEK',
    'SCHEDULED_DEPARTURE_HR',
    'SCHEDULED_ARRIVAL_HR',
    'SCHEDULED_TIME_HR',
    'DISTANCE'
]

# pre processamento
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# algoritmo LogisticRegression
lr = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))
])

# dividir os dados entre treino (70%) e teste (30%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=23,
    stratify=y  
)

# treinar modelo
lr.fit(X_train, y_train)

# testando thresholds
thresholds = [0.3, 0.4, 0.5, 0.6]

precisions = []
recalls = []

y_prob = lr.predict_proba(X_test)[:, 1]

for t in thresholds:
    y_pred = (y_prob > t).astype(int)
    precisions.append(precision_score(y_test, y_pred))
    recalls.append(recall_score(y_test, y_pred))

plt.figure()
plt.plot(thresholds, precisions, label='Precision')
plt.plot(thresholds, recalls, label='Recall')
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision vs Recall por threshold")
plt.legend()
plt.grid(True)
plt.show()

# modelo final com o threshold escolhido de 0.4
y_pred = (y_prob > 0.4).astype(int)

# métricas
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 score:", f1_score(y_test, y_pred))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Matriz de confusão")
plt.show()

