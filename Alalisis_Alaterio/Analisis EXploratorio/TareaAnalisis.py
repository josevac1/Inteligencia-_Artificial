# ==========================================
# 0. INSTALACIÓN DE LIBRERÍAS (Solo si es necesario)
# ==========================================


import kagglehub
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 1. CARGA DE DATOS (Descarga Automática)
# ==========================================
# Descarga el conjunto de datos de interés desde Kaggle
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
archivo = os.path.join(path, "creditcard.csv")
df = pd.read_csv(archivo)

print("✓ Dataset importado correctamente.")
print(f"Tamaño: {df.shape[0]} observaciones y {df.shape[1]} variables.")

# ==========================================
# 2. EXPLORACIÓN INICIAL Y CALIDAD
# ==========================================
# Verificar tipos de variables y calidad (nulos)
print("\n--- Tipos de Variables ---")
print(df.dtypes.head()) 

print("\n--- Verificación de Valores Faltantes ---")
print(f"Total de valores nulos: {df.isnull().sum().max()}")

# ==========================================
# 3. RESUMEN ESTADÍSTICO
# ==========================================
print("\n--- Estadísticas Descriptivas (Time & Amount) ---")
print(df[['Time', 'Amount']].describe())
# Conclusión: La alta desviación estándar en Amount indica gran variabilidad.

# ==========================================
# 4. ANÁLISIS DE VARIABLES CATEGÓRICAS
# ==========================================
# Convertimos Amount a categoría para análisis de frecuencia
df['Amount_cat'] = pd.cut(df['Amount'], bins=5, labels=['Muy Bajo','Bajo','Medio','Alto','Muy Alto'])

print("\n--- Frecuencia por Niveles de Monto ---")
print(df['Amount_cat'].value_counts())

# FIGURA: Gráfico de barras de categorías
plt.figure(figsize=(7, 4))
sns.countplot(x='Amount_cat', data=df, palette='Blues')
plt.title("Análisis de Frecuencia: Categorías de Monto")
plt.show()

# ==========================================
# 5. VISUALIZACIÓN DE DATOS (EDA)
# ==========================================
# FIGURA 1: Distribución de la variable objetivo Y (Class)
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=df, palette='viridis')
plt.title("Distribución de Clases (0 = Normal, 1 = Fraude)")
plt.show()

# FIGURA 2: Relación Amount vs Time (Dispersión)
plt.figure(figsize=(8, 5))
plt.scatter(df[df.Class == 0]['Time'], df[df.Class == 0]['Amount'], c='g', s=1, label='Normal', alpha=0.1)
plt.scatter(df[df.Class == 1]['Time'], df[df.Class == 1]['Amount'], c='r', s=10, label='Fraude')
plt.title("Relación Monto vs Tiempo de Transacción")
plt.legend()
plt.show()

# ==========================================
# 6. ANÁLISIS DE CORRELACIÓN
# ==========================================
# Mapa de calor personalizado (Triángulo inferior)
cr = df.drop(['Amount_cat'], axis=1).corr()
mask = np.zeros_like(cr)
mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cr, mask=mask, square=True, linewidths=.5, cmap="YlGnBu")
    plt.title("Mapa de Correlación de Atributos")
    plt.show()

# ==========================================
# 7. ANÁLISIS DE OUTLIERS
# ==========================================
# Visualización 3D para comprender valores atípicos
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Muestra de datos para claridad visual
normales_sample = df[df['Class'] == 0].sample(2000, random_state=42)
fraude_cases = df[df['Class'] == 1]

ax.scatter(normales_sample['V1'], normales_sample['V2'], normales_sample['Amount'], c='green', s=10, alpha=0.3)
ax.scatter(fraude_cases['V1'], fraude_cases['V2'], fraude_cases['Amount'], c='red', s=20, marker='x')
ax.set_title("Identificación Visual de Outliers (Fraude)")
plt.show()

# ==========================================
# 8. MODELADO Y GENERACIÓN DE HIPÓTESIS
# ==========================================
# Segmentación para entrenamiento
X = df.drop(['Class', 'Amount_cat'], axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Escalado
scaler = StandardScaler()
X_train[['Amount', 'Time']] = scaler.fit_transform(X_train[['Amount', 'Time']])
X_test[['Amount', 'Time']] = scaler.transform(X_test[['Amount', 'Time']])

# Modelo (Priorizando Recall)
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)

# Evaluación
y_scores = model.predict_proba(X_test)[:, 1]
precision, recall, _ = precision_recall_curve(y_test, y_scores)

print("\n--- Reporte de Clasificación ---")
print(classification_report(y_test, model.predict(X_test)))

# Conclusión Final: AUPRC
print(f"Resultado final AUPRC: {auc(recall, precision):.4f}")