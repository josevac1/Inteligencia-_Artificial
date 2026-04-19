import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 1. CARGA DE DATOS Y CONFIGURACIÓN DE RUTA
# ==========================================
# Esto asegura que busque el archivo en la carpeta donde tienes guardado el script .py
ruta_directorio = os.path.dirname(os.path.abspath(__file__))
ruta_archivo = os.path.join(ruta_directorio, "creditcard.csv")

try:
    df = pd.read_csv(ruta_archivo)
    print("✓ Dataset cargado correctamente.")
    print(f"Estructura: {df.shape[0]} filas y {df.shape[1]} columnas.")
except FileNotFoundError:
    print(f" Error: No se encontró 'creditcard.csv' en {ruta_directorio}")
    exit()

# Función para contar categorías (de tu código original)
def descripcionCantidadCategorias(dataframe, columnas):
    for variable in columnas:
        cantidad = dataframe[variable].nunique()
        print(f'Cantidad de categorías en {variable}:', cantidad)

# ==========================================
# 2. ANÁLISIS DE VARIABLES CATEGÓRICAS
# ==========================================
# Crear variables categóricas (ordinales) a partir de datos numéricos
df['Amount_cat'] = pd.cut(df['Amount'], bins=5, labels=['Muy Bajo','Bajo','Medio','Alto','Muy Alto'])
df['Time_cat'] = pd.cut(df['Time'], bins=5, labels=['Muy Bajo','Bajo','Medio','Alto','Muy Alto'])

print("\n--- Descripción de variables categóricas creadas ---")
descripcionCantidadCategorias(df, ['Amount_cat', 'Time_cat'])
print(df[['Amount_cat', 'Time_cat']].head())

# ==========================================
# 3. VISUALIZACIONES
# ==========================================

# FIGURA 1: Distribución de clases (0 y 1)
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=df, palette='viridis')
plt.title("Distribución de clases (0 = Normal, 1 = Fraude)")
plt.xlabel("Clase")
plt.ylabel("Cantidad de transacciones")
plt.show()

# FIGURA 2: Categorías de Amount
plt.figure(figsize=(7, 4))
sns.countplot(x='Amount_cat', data=df, palette='Blues')
plt.title("Frecuencia por Niveles de Monto")
plt.show()

# FIGURA 3: Mapa de Calor Personalizado (Estilo exacto de tu guía)
# Eliminamos las columnas no numéricas para el cálculo de correlación
cr = df.drop(['Amount_cat', 'Time_cat'], axis=1).corr()
mask = np.zeros_like(cr)
mask[np.triu_indices_from(mask)] = True # Máscara para la región superior

with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(12, 10))
    ax = sns.heatmap(cr, mask=mask, square=True, linewidths=.5, cmap="YlGnBu", annot=False)
    plt.title("Mapa de Calor: Correlación de Atributos (Triángulo Inferior)")
    plt.savefig('attribute_correlations.png') # Guardado de imagen automático
    plt.show()

# FIGURA 4: Visualización 3D (Análisis de Outliers / Clases)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Tomamos una muestra de normales para no saturar el gráfico, pero mantenemos todos los fraudes
normales = df[df['Class'] == 0].sample(2000, random_state=42)
fraudes = df[df['Class'] == 1]

# CORRECCIÓN DE ERROR: 'green' con dos 'e'
ax.scatter(normales['V1'], normales['V2'], normales['Amount'], c='green', s=10, label='Normal', alpha=0.4)
ax.scatter(fraudes['V1'], fraudes['V2'], fraudes['Amount'], c='red', s=15, label='Fraude', marker='x')

ax.set_title("Detección de outliers y fraudes en espacio 3D")
ax.set_xlabel('V1')
ax.set_ylabel('V2')
ax.set_zlabel('Monto')
ax.legend()
plt.show()

# ==========================================
# 4. PREPROCESAMIENTO PARA MODELADO
# ==========================================
# Limpieza: eliminamos las columnas temporales de visualización antes de entrenar
X = df.drop(['Class', 'Amount_cat', 'Time_cat'], axis=1)
y = df['Class']

# División de datos (Estratificada por el desbalance extremo)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Escalado: StandardScaler aplicado después del split para evitar Data Leakage
scaler = StandardScaler()
X_train[['Amount', 'Time']] = scaler.fit_transform(X_train[['Amount', 'Time']])
X_test[['Amount', 'Time']] = scaler.transform(X_test[['Amount', 'Time']])

# ==========================================
# 5. MODELO Y EVALUACIÓN
# ==========================================
# Usamos class_weight='balanced' para compensar que hay muy pocos casos de fraude
model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Predicciones y probabilidades
y_pred = model.predict(X_test)
y_scores = model.predict_proba(X_test)[:, 1]

print("\n" + "="*40)
print(" RESULTADOS DEL MODELO")
print("="*40)
print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))

# FIGURA 5: Curva Precision-Recall (La métrica recomendada en tu guía para datos desbalanceados)
precision, recall, _ = precision_recall_curve(y_test, y_scores)
auprc = auc(recall, precision)

plt.figure(figsize=(7, 5))
plt.plot(recall, precision, color='darkblue', lw=2, label=f'AUPRC = {auprc:.4f}')
plt.fill_between(recall, precision, alpha=0.2, color='skyblue')
plt.title("Curva Precision-Recall")
plt.xlabel("Recall (Capacidad de encontrar fraudes)")
plt.ylabel("Precision (Fiabilidad de la predicción)")
plt.legend(loc="upper right")
plt.grid(alpha=0.3)
plt.show()

print(f"\nÁrea Final bajo la curva (AUPRC): {auprc:.4f}")