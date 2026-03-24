from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
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
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numeric_features)  
    ]
)

# algoritmo DecisionTreeClassifier
dtr = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(
        max_depth=15,           
        min_samples_split=15,   
        class_weight = 'balanced',
        random_state=23
    ))
])

# dividir os dados entre treino (70%) e teste (30%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=23,
    stratify=y
)

# treinar modelo
dtr.fit(X_train, y_train)

# realizar previsoes
y_pred = dtr.predict(X_test)

# métricas
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 score:", f1_score(y_test, y_pred))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Matriz de confusão")
plt.show()