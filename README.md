# Análisis de Clasificación de Diabetes

## Descripción del Proyecto

Este proyecto realiza un análisis de clasificación para predecir la presencia de diabetes en pacientes basándose en un conjunto de datos de características médicas. El objetivo es implementar un modelo de clasificación usando el algoritmo RandomForestClassifier y evaluar su desempeño.

## Conjunto de Datos

El conjunto de datos utilizado se encuentra en el archivo `diabetes_indiana.csv`. Este archivo contiene las siguientes características:

- **Feature1**: [Descripción de Feature1]
- **Feature2**: [Descripción de Feature2]
- **Feature3**: [Descripción de Feature3]
- **Feature4**: [Descripción de Feature4]
- **Feature5**: [Descripción de Feature5]
- **Feature6**: [Descripción de Feature6]
- **Feature7**: [Descripción de Feature7]
- **Feature8**: [Descripción de Feature8]
- **Outcome**: Variable objetivo que indica la presencia de diabetes (1 para positivo, 0 para negativo)

## Pasos del Análisis

1. **Carga del Conjunto de Datos**: El archivo CSV se carga y se asignan nombres a las columnas.
2. **Preparación de Datos**:
   - Se dividen los datos en variables predictoras (`X`) y variable objetivo (`y`).
   - Se dividen en conjuntos de entrenamiento y prueba.
3. **Estandarización**: Se estandarizan los datos para mejorar el rendimiento del modelo.
4. **Creación y Ajuste del Modelo**:
   - Se utiliza el clasificador `RandomForestClassifier`.
   - El modelo se ajusta a los datos de entrenamiento.
5. **Evaluación del Modelo**:
   - Se imprimen la matriz de confusión y el informe de clasificación.
   - Se grafica la matriz de confusión usando `seaborn`.

## Código

El código se encuentra en el archivo `diabetes_classification.py`. Aquí está el código utilizado:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el conjunto de datos
df = pd.read_csv('diabetes_indiana.csv', header=None)

# Asignar nombres a las columnas si el archivo no tiene encabezado
column_names = ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5', 
                'Feature6', 'Feature7', 'Feature8', 'Outcome']
df.columns = column_names

# Verificar las primeras filas del DataFrame para asegurarse de que se cargó correctamente
print(df.head())

# Dividir el conjunto de datos en variables predictoras (X) y variable objetivo (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Estandarizar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crear el modelo de clasificación
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Ajustar el modelo
model.fit(X_train, y_train)

# Hacer predicciones
y_pred = model.predict(X_test)

# Evaluar el modelo
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Graficar la matriz de confusión
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
