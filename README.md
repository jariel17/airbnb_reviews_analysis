# Análisis de Opiniones y Características de Listados de Airbnb en Ciudad México

## Objetivo
El objetivo de este proyecto es analizar las opiniones y características de los listados de Airbnb en México para obtener información valiosa sobre las preferencias de los usuarios y las tendencias del mercado. Se evaluan las características de los listados, las opiniones de los usuarios.

## Pasos para usar el proyecto

1. **Instalar dependencias**:
   - Crea un entorno virtual para el proyecto:
     ```bash
     python -m venv env
     ```
   - Activa el entorno virtual ejecutando el siguiente comando en la terminal:
     ```bash
     source env/bin/activate
     ```
   - Instala las dependencias necesarias ejecutando:
     ```bash
     pip install -r requirements.txt
     ```

2. **Explorar los datos**:
   - Se proporcionará un script para descargar los datos.
   - Archivos disponibles:
     - `listings.csv`: Contiene información sobre los listados.
     - `reviews.csv`: Contiene las opiniones de los usuarios.

3. **Ejecutar el análisis**:
   - Abre el archivo `airbnb.ipynb` en Jupyter Notebook para ejecutar el análisis y visualizar los resultados.

4. **Contribuir**:
   - Si deseas contribuir al proyecto, realiza un fork del repositorio y envía tus cambios mediante un pull request.

## Desarrollo del Proyecto
En esta sección se documentarán los pasos realizados durante el desarrollo del proyecto.

1. Fomulación del problema.
Se define el problema como una clasificación binaria ya que da más valor para los interesados el entender si un listado es bueno o malo, en lugar de una clasificación ordinal. Se define como bueno si la puntuación es mayor a 4.5 y malo si es menor a 4.5. Existen varios rating en el dataset, pero tomaremos en cuenta el `review_scores_rating`, ya que es la calificación general del listado.
Algunas de las preguntas que se intentan responder son:
   - ¿Qué características de los listados están asociadas con una alta puntuación?
   - ¿Cuáles son las características estructuradas que más se correlacionan con una calificación alta?
   - ¿Cómo varían las opiniones de los usuarios según la ubicación y el tipo de alojamiento?
   - ¿Qué factores influyen en la decisión de los usuarios al elegir un listado?
   - ¿Existen patrones en el texto de los comentarios que indiquen problemas frecuentes (por ejemplo, “limpieza” o “ruido”) asociados a calificaciones bajas?