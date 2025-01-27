
#################################################################################
#               UNIVERSIDAD NACIONAL DE SAN MARTIN (UNSAM)

#                 MATEMATICA III - 1ER CUATRIMESTRE 2023
#                       TRABAJO PRACTICO FINAL 

#                       ALUMNA: MECOZZI TAMARA E.

#El TP final consta de un trabajo de análisis (individual) sobre un dataset, a elección.

#Fecha de entrega : 20/06/2023


#################################################################################
"""
Archivo utilizado "recaudacion-impositiva.csv"

Analisis de la Recaudación impositiva  de la Ciudad Autonoma de Buenos Aires

Compuesta por nueve columnas:

    - Periodo (sera modificada para el analisis)
    - Total
    - Impuesto sobre los IIBB
    - Alumbrado barrido y limpieza
    - Impuesto patente 
    - Impuesto al sello
    - Plan facilidad de pagos 
    - Contribucion porr publicidad
    - Gravamenes varios
    

#Etapa 1 ⇒ Detectar el problema: Qué queremos estimar o predecir?

Estimar con RLM:
       La predicción de la recaudación impositiva total en la Ciudad Autónoma de Buenos Aires
                           (CABA) de los ultimos 15 años.

Estimar con RLS:
    La relación lineal entre el Impuesto sobre los IIBB y la recaudación impositiva 
                      total en la Ciudad Autónoma de Buenos Aires (CABA) 
                                durante los últimos 15 años.
            
"""

#Librerias necesarias para el manejo y analisis del dataset

import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from scipy import stats as st




#Importacion de los datos

archivo = "recaudacion-impositiva.csv"
data = pd.read_csv(archivo, sep=';')
print( '\n', data,'\n' )

#Cantidad de filas y columnas del dataset 

print("El dataset propuesto al analisis tiene {} filas por {} columnas que corresponden a la informacion de las recaudaciones impositivas".format(len(data),len(data.columns)),)
print('\n')

data.info()

print("\nEl dataset tiene 8 variables numericas y una categorica\n")

#Analisis de las varibales cadena de caracteres

"""
Para cotinuar con el analisis, la columna periodo la separo en año y mes. 
Agrupo por año y elimino la columna mes, para evitar tener meses repetidos.
"""
# Separar la columna "periodo" en "mes" y "año"
data[['mes', 'año']] = data['periodo'].str.split('-', expand=True)

# Eliminar la columna "periodo"
data = data.drop('periodo', axis=1)

# Agrupar y sumar las filas por año
data = data.groupby('año').sum().reset_index()

# Eliminar la columna "mes" del DataFrame resultante
data = data.drop('mes', axis=1, errors='ignore')

# Imprimir el DataFrame resultante con lo obtenido
print(data)
#Sera nesario seguir trabajando esa columna 
# Convertir la columna "año" a tipo numérico
data['año'] = pd.to_numeric(data['año'], errors='coerce')

# Agregar prefijo "20" a los años de dos dígitos
data['año'] = data['año'].apply(lambda x: 2000 + x if x <= 22 else x)

######## Aplicado los filtros necesarios para quedarnos con los ultimos 15 años#####

# Filtrar los años del 2008 al 2022 inclusive
data = data[(data['año'] >= 2008) & (data['año'] <= 2022)]
print(data)

#  filas y columnas necesarias del dataset

print("Por lo tanto, el dataset queda conformado con {} filas por {} columnas que corresponden a la informacion de las recaudaciones impositivas".format(len(data),len(data.columns)))
print('\n')

data.info()
print('\n')

print ("La variable Año tiene los siguientes valores: {}".format(pd.unique(data.año)))
data.total.value_counts()
print('\n')

# segunda columna 
print ("La variable total tiene los siguientes valores: {}".format(pd.unique(data.total)))
data.total.value_counts()
print('\n')

#tercer columna
print ("La variable Impuesto sobre los IIBB tiene los siguientes valores:{}".format(pd.unique(data.impuestos_sobre_ingresos_brutos)))
data.impuestos_sobre_ingresos_brutos.value_counts()
print('\n')

#cuarta columna
print ("La variable Alumbrado barrido y limpieza tiene los sguientes valores:{}".format(pd.unique(data.alumbrado_barrido_limpieza)))
data.alumbrado_barrido_limpieza.value_counts()
print('\n')

#quinta columna 
print ("La variable Impuesto patentes tiene los siguientes valores: {}".format(pd.unique(data.impuesto_patentes)))
data.impuesto_patentes.value_counts()
print('\n')

#sexta columna 
print ("La variable Impuesto al sello tiene los siguientes valores: {}".format(pd.unique(data.impuesto_sellos)))
data.impuesto_sellos.value_counts()
print('\n')

#septima columna 
print ("La variable plan facilidad de pagos tiene los siguientes valores: {}".format(pd.unique(data.plan_facilidades_pago)))
data.plan_facilidades_pago.value_counts()
print('\n')
#octava columna 
print ("La variable contribuciones por publicidad de pagos tiene los siguientes valores: {}".format(pd.unique(data.contribucion_por_publicidad)))
data.contribucion_por_publicidad.value_counts()
print('\n')
#novena columna 
print ("La variable gravamenes varios tiene los siguientes valores: {}".format(pd.unique(data.gravamenes_varios)))
data.gravamenes_varios.value_counts()
print('\n')

#Analisis del coeficiente de correlación:

total_coef = data.corr(numeric_only=True).total
print(total_coef.abs().sort_values(ascending=False))

"""El coeficiente permite identificar las variables que tienen una mayor o menor correlación 
   con la columna "Total", puede ser útil para entender las relaciones entre las
    variables y su influencia en la recaudación impositiva total."""

# Separacion de variable independiente de la variable dependiente 
###################################################################
# Variables independientes :  Impuesto sobre los IIBB
                           #  Alumbrado barrido y limpieza
                           #  Impuesto patente 
                           #  Impuesto al sello
                           #  Plan facilidad de pagos 
                           #  Contribucion porr publicidad
                           #  Gravamenes varios
                           
# División del dataset: datos de entrenamiento y validación
# Se divide el conjunto de datos: 
#  un 80% para entrenamiento (x_train, y_train) 
#  un 20% para prueba (x_test, y_test)

# Variables predictoras (X)
x = data.iloc[:, [0, 2, 3, 4, 5, 6, 8]].values
print(x)

# Variable objetivo (y)
y = data.iloc[:, [1]].values
print(y)                           
                           
#Analisis si existen valores NaN
#Detectar los valores desconocidos 

print(data.isnull().values.any())

#Devuelve True a las variables que tengan un dato desconocido 
print(data.isnull().any())
#Indentifico la cantidad de valores desconocidos (NaN)
print(data.isnull().sum().sum())

print("No hay valores faltante o desconocidos")

#Si existian valores NaN, usando SKLEARN permite sustituir valores nulos por otros valores
"""
la ausencia de valores NaN en el dataset es favorable, ya que permite la integridad de 
los datos generando un análisis más sólido y confiable de la recaudación impositiva 
en la Ciudad Autónoma de Buenos Aires."""


# Division del dataset: datos de entrenamiento y validacion test
# Se divide el conjunto de datos: 
   #  un 80% para entrenamiento (x_train, y_train) 
   #  un 20% para prueba (x_test, y_test)

# Dividir los datos en conjuntos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
print(x_train)

# Estandarización de los datos de las variables predictoras
# Se crea una instancia del escalador StandardScaler
sc_x = StandardScaler()

# Se ajusta y transforma los datos de entrenamiento (x_train) utilizando el escalador

x_train = sc_x.fit_transform(x_train)
print("\nLos datos de entrenamiento: \n\n",x_train)

# Se transforman los datos de prueba (x_test) utilizando el escalador previamente ajustado

x_test = sc_x.transform(x_test)
print("\nLos datos para testeo: \n\n",x_test)

###################################################################################
                    
                       ### REGRESION LINEAL MULTIPLE###
                       
#  El objetivo principal es obtener un modelo de regresión que pueda predecir con precisión
# la recaudación impositiva total en base a las variaciones en las variables predictoras.

  
  
###########################################################################################
print("Regresión lineal multiple")

# Informacion de los datos

print("Resumen estadístico del dataset:")
print(data.describe())

# Cálculo de los coeficientes de correlación del total de impuestos con los demás atributos
total_coef = data.corr(numeric_only=True).total
print("Coeficientes de correlación del total de impuestos:")
print(total_coef.abs().sort_values(ascending=False))

# Calcular la matriz de correlación
# Descarto la variable año por mas que sea numerica, ya que representa diferentes períodos
#de tiempo y su presencia en la matriz de correlación podría generar valores de correlación
#engañosos o poco significativos.
data_num = data.drop(columns=['año'])
corr_matrix = data_num.corr()

# Crear una figura y un eje para el gráfico
fig, ax = plt.subplots(figsize=(6, 6))

# Graficar la matriz de correlación utilizando seaborn
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, square=True, ax=ax)

# Definir las etiquetas de los ejes x e y
ax.set_xticklabels(data_num.columns, rotation=45, ha='right')
ax.set_yticklabels(data_num.columns, rotation=0)

# Agregar título y mostrar el gráfico
plt.title('Matriz de correlación')
plt.show()

# Crear una tabla con los coeficientes de correlación
corr_table = corr_matrix.unstack().sort_values(ascending=False).reset_index()
corr_table.columns = ['Variable 1', 'Variable 2', 'Coeficiente de correlación']

"""
La matriz de correlación muestra que el total de impuestos está fuertemente influenciado
 por variables como el impuesto sobre los ingresos brutos, el impuesto de sellos,
 el impuesto de patentes y otras contribuciones.
""" 
# Crear figura y ejes
fig, ax = plt.subplots(figsize=(8, 6))

# Graficar los datos
ax.hist(data['total'], align='left', alpha=0.5, bins=100)

# Establecer etiquetas de los ejes
ax.set_xlabel('Total de impuestos')
ax.set_ylabel('Frecuencia')
ax.set_title('Distribución del Total de Impuestos')

# Agregar una leyenda para explicar las columnas
ax.text(0.95, 0.95, 'Columna: Rango de valores del total de impuestos',
        verticalalignment='top', horizontalalignment='right',
        transform=ax.transAxes, fontsize=10, bbox={'facecolor': 'white', 'alpha': 0.5})

# Mostrar el gráfico
plt.show()

"""
Existe una concentración de casos en los dos primeros rangos de valores del total de 
impuestos, mientras que los demás rangos tienen una cantidad similar de observaciones dispersas """


#Entrenar modelo 
#Considerando la division del dataset: datos de entrenamiento y validacion testdef get_coefficients(x_train, y_train):




#Los datos en conjuntos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
print(x_train)

# Estandarización de los datos de las variables predictoras
# Se crea una instancia del escalador StandardScaler
sc_x = StandardScaler()

# Se ajusta y transforma los datos de entrenamiento (x_train) utilizando el escalador
x_train = sc_x.fit_transform(x_train)
print("\nLos datos de entrenamiento: \n\n", x_train)

# Se transforman los datos de prueba (x_test) utilizando el escalador previamente ajustado
x_test = sc_x.transform(x_test)
print("\nLos datos para testeo: \n\n", x_test)

# Crear una instancia del modelo de regresión lineal
regressor = LinearRegression()

# Entrenar el modelo con los datos de entrenamiento
regressor.fit(x_train, y_train)

# Obtener los coeficientes del modelo
coefficients = regressor.coef_

# Nombre de las variables predictoras
predictor_vars = ['impuestos_sobre_ingresos_brutos', 'alumbrado_barrido_limpieza', 'impuesto_patentes', 'impuesto_sellos', 'plan_facilidades_pago', 'contribucion_por_publicidad', 'gravamenes_varios']

# Imprimir los coeficientes junto con el nombre de cada variable
print("Coeficientes del modelo:")
for var, coef in zip(predictor_vars, coefficients[0]):
    print(f"{var}: {coef}")

# Verificar las dimensiones de los datos de prueba
print(x_test.shape)  # Asegúrate de que coincida con la forma utilizada durante el entrenamiento

# Realizar la predicción sobre los datos de prueba
y_pred = regressor.predict(x_test)

# Convertir y_test y y_pred en arrays unidimensionales
y_test = y_test.flatten()
y_pred = y_pred.flatten()

# Crear el DataFrame para comparar los valores reales y los valores predichos
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# Calcular las diferencias entre los valores reales y los valores predichos
comparison['Difference'] = comparison['Actual'] - comparison['Predicted']

# Crear el gráfico de barras
ax = comparison.head(10).plot(kind='bar', y='Difference', figsize=(10, 8), color=['darkblue' if val >= 0 else 'darkred' for val in comparison['Difference']])

# Configurar las etiquetas en el eje x
ax.set_xticklabels(comparison.index[:10])

# Configurar el título del gráfico
plt.title('Diferencias entre valores reales y predichos')

# Configurar la etiqueta del eje y
plt.ylabel('Diferencia')

# Configurar la cuadrícula
plt.grid(which='major', linestyle='-', linewidth='0.5', color='darkgreen')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')


# Mostrar el gráfico
plt.show()

#El gráfico de barras mostrando las diferencias entre los valores reales y los de la predicción
#Las barras en crecimiento muestran las diferencias entre los valores reales y los valores predichos están aumentando gradualmente.

# Calcular y mostrar las métricas de evaluación del modelo
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


####################################################################################

                #### REGRESION LINEAL SIMPLE #####
                
"""
 El objetivo es determinar si existe una relación lineal significativa 
 entre el Impuesto sobre los IIBB y la recaudación total, y utilizar esta relación 
 para predecir la recaudación impositiva en base a la variación en el Impuesto sobre los IIBB            
"""                
####################################################################################
print("Regresión lineal simple")
# Cálculo de la media del total de impuestos para cada valor del impuesto a los ingresos brutos
impuestos_mean = data.groupby(by="impuestos_sobre_ingresos_brutos").total.mean().round(2)

# Gráfico de barras mejorado
fig, ax = plt.subplots(figsize=(8, 6))  #tamaño de la figura
impuestos_mean.plot(kind='bar', ax=ax)  # Utiliza el objeto de los ejes 'ax' para dibujar el gráfico
ax.set_xlabel("Impuesto sobre los Ingresos Brutos") 
ax.set_ylabel("Total de Impuestos ($)") 
ax.set_title("Relación entre el Impuesto a los Ingresos Brutos y el Total de Impuestos")  
plt.xticks(rotation=45)  # Rota las etiquetas del eje x
plt.tight_layout()  # Ajusta el espaciado entre elementos 
plt.show()

"""
El grafico muestra como se incrementa exponencialmente,es decir, proporciona
   información relevante sobre qué variables están influyendo de manera más 
   significativa en el crecimiento de la recaudación impositiva. 
"""

print("Esta función devuelve un resumen estadístico que incluye la media, la mediana, el valor mínimo y máximo, la desviación estándar y los cuartiles.", data.describe())

print("Obtengo una tupla que representa la dimensionalidad del dataset", data.shape)

print("max total: ", data['total'].max())
print("min total: ", data['total'].min())
print("max impuestos_sobre_ingresos_brutos: ", data['impuestos_sobre_ingresos_brutos'].max())
print("min impuestos_sobre_ingresos_brutos: ", data['impuestos_sobre_ingresos_brutos'].min())

# Realizo un grafico los puntos de datos en un diagrama en dos dimensiones para ilustrar el dataset 
data.plot(x='impuestos_sobre_ingresos_brutos', y='total', style='o')
plt.title('Impuestos sobre ingresos brutos vs Total')
plt.xlabel('Impuestos sobre ingresos brutos')
plt.ylabel('Total')
plt.show()


"""
Este gráfico muestra la superposición inicial de puntos sugiere que en los rangos 
 más bajos del impuesto sobre los ingresos brutos, los totales de impuestos recaudados
 son similares. A medida que el impuesto sobre los ingresos brutos aumenta, 
 los totales de impuestos también aumentan, pero con una mayor variabilidad,
 lo que se refleja en la dispersión de los puntos en el extremo derecho del gráfico
"""


#Considerando la division del dataset: datos de entrenamiento y validacion test

regressor = LinearRegression() 
regressor.fit(x_train, y_train)

print("Para obtener el interceptor:")

print(regressor.intercept_)

print("Para obtener la pendiente:")

print(regressor.coef_)

#Cada coeficiente representa el cambio esperado en 'total' cuando la variable predictora correspondiente aumenta 

#Analisis de con que  precisión el algoritmo predice la puntuación porcentual.
y_pred = regressor.predict(x_test)
df_aux = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(df_aux)

# Convertir los arrays en listas
x_test_list = x_test.flatten().tolist()
y_test_list = y_test.flatten().tolist()
y_pred_list = y_pred.flatten().tolist()

# Verificar y ajustar la longitud de las listas si es necesario
if len(x_test_list) > len(y_test_list):
    x_test_list = x_test_list[:len(y_test_list)]
elif len(y_test_list) > len(x_test_list):
    y_test_list = y_test_list[:len(x_test_list)]
# Gráfico de dispersión de los valores de prueba (x_test) y reales (y_test)
plt.scatter(x_test_list, y_test_list, color='blue', label='Valores reales')

# Gráfico de línea para los valores de prueba (x_test) y predichos (y_pred)
plt.plot(x_test_list, y_pred_list, color='red', linewidth=2, label='Valores predichos')
# Agregar etiquetas en el gráfico
plt.text(0.1, 0.9, 'Valores reales', color='blue', transform=plt.gca().transAxes)
plt.text(0.1, 0.85, 'Valores predichos', color='red', transform=plt.gca().transAxes)

# Configuración adicional del gráfico
plt.xlabel('x_test')
plt.ylabel('Valores')
plt.title('Comparación de Valores Reales y Predichos')

# Mostrar el gráfico
plt.show()

print('Error Absoluto Medio:', metrics.mean_absolute_error(y_test, y_pred))
print('Error Cuadrático Medio:', metrics.mean_squared_error(y_test, y_pred))
print('Raíz del error cuadrático medio:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#Los valores obtenidos en las métricas de evaluación proporcionan información sobre el rendimiento y la precisión del modelo de regresión lineal.

print("\nEl modelo de regresión lineal muestra cierta capacidad para capturar la tendencia general de los datos, los errores obtenidos se pueden interpretar de que existe una cantidad significativa de variabilidad y desviación entre las predicciones y los valores reales. Es importante considerar en que contexto se aplica. \n")

""" Apreciación personal de los resultados:
    El aumento en el impuesto a los ingresos brutos (IIBB) en la Ciudad Autónoma de Buenos Aires
 (CABA), podría estar correlacionado con el número de monotributistas y la disminución
 de empleados en relación de dependencia, ya que el impuesto a los IIBB es un costo adicional
 que puede desincentivar la contratación formal de personal y fomentar la informalidad
 o el autoempleo.Este analisis podria ser util para evaluar el trabajo independiente """
