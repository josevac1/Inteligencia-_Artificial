import numpy as np
import pandas as pd
import copy
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Carga de datos
archivo_excel = 'Libro1.xlsx'
df_original = pd.read_excel(archivo_excel, sheet_name='Hoja1')

print('Cantidad de observaciones (clientes):', df_original.shape[0])
print('Cantidad de variables:', df_original.shape[1])
print('Forma del dataset:', df_original.shape)
print(df_original.head())

# Creamos una copia de seguridad para trabajar
df_procesado = copy.deepcopy(df_original)

# --- LIMPIEZA DE DATOS ---
df_procesado['pais'] = df_procesado['pais'].str.strip().str.lower()
df_procesado['NivelSatifaccion'] = df_procesado['NivelSatifaccion'].astype(str).str.strip().str.lower()

# 2. SEPARACIÓN DE VARIABLES
# Guardamos la variable de salida original por si la necesitamos después
Y_original = df_procesado[['NivelSatifaccion']]

# Definición de variables
vars_nominales = ['Sexo', 'pais']
vars_numericas = ['edad']
vars_ordinales = ['NivelSatifaccion']

# 3. Función para predecir cantidad de columnas
def analizar_variables(dataframe, vars_nom, vars_ord):
    total_columnas_originales = len(dataframe.columns)
    total_vars_nominales = len(vars_nom)
    total_nuevas_binarias = 0
    
    for variable in vars_nom:
        cantidad_categorias = dataframe[variable].nunique()
        total_nuevas_binarias += cantidad_categorias
        print(f'Categorías en variable nominal "{variable}": {cantidad_categorias}')
        
    print('Nuevas columnas binarias a crear:', total_nuevas_binarias)
    
    # Cálculo final
    total_columnas_final = total_columnas_originales - total_vars_nominales + total_nuevas_binarias
    return total_columnas_final

# Ejecutamos la función y GUARDAMOS el número
total_columnas_esperadas = analizar_variables(df_procesado, vars_nominales, vars_ordinales)
print('Total de columnas tras la transformación:', total_columnas_esperadas)

# 4. Definición de Transformadores

mi_orden_logico = [['no me gusta', 'neutral', 'me gusta']]
# Categóricos
trans_ordinal = Pipeline(steps=[('ordinal', OrdinalEncoder(categories=mi_orden_logico))])
trans_nominal = Pipeline(steps=[('one_hot', OneHotEncoder(sparse_output=False, handle_unknown="ignore"))])

preprocesador_categorico = ColumnTransformer(transformers=[
    ('cat_ord', trans_ordinal, vars_ordinales),
    ('cat_nom', trans_nominal, vars_nominales)
], remainder='passthrough', n_jobs=-1)

# --- CAMBIO A ESTANDARIZACIÓN (StandardScaler) ---
trans_scaler = Pipeline(steps=[('scaler', StandardScaler())])
preprocesador_scaler = ColumnTransformer(transformers=[
    ('trans_scaler', trans_scaler, list(range(total_columnas_esperadas)))
], remainder='passthrough')

# 5. Construcción del Pipeline Maestro
pipe = Pipeline(steps=[
    ('prep_categorico', preprocesador_categorico), 
    ('prep_escalado', preprocesador_scaler) # <- Usamos el preprocesador con StandardScaler
])

# 6. Ejecución del Pipeline
X_transformado = pipe.fit_transform(df_procesado)
print('\n********** Pipeline aplicado **********')

# 7. Reconstrucción de nombres de columnas
nombres_columnas_finales = []

if len(vars_ordinales) != 0:
    nombres_columnas_finales.extend(vars_ordinales)

if len(vars_nominales) != 0:
    nombres_nuevas_vars = pipe.named_steps['prep_categorico'].transformers_[1][1].named_steps['one_hot'].get_feature_names_out(vars_nominales)
    nombres_columnas_finales.extend(nombres_nuevas_vars)

if len(vars_numericas) != 0:
    nombres_columnas_finales.extend(vars_numericas)

print('********** Lista de variables reconstruidas:')
print(nombres_columnas_finales)

# Reconstruimos el DataFrame con Pandas
df_final = pd.DataFrame(data=X_transformado, columns=nombres_columnas_finales)

# --- 8. IMPLEMENTACIÓN DEL CONCAT ---
# Aquí volvemos a unir la variable Y original al final del dataset transformado
df_final_con_etiquetas = pd.concat([df_final, Y_original.reset_index(drop=True)], axis=1)

# Guardamos el archivo (¡Ojo! Asegúrate de exportar el que tiene las etiquetas unidas)
df_final_con_etiquetas.to_excel('Dataset_Transformado_Estandarizado.xlsx', index=False)

print("\nVista previa de los datos Estandarizados:")
print(df_final_con_etiquetas.head(6))