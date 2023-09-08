import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
import folium

# Descargar el lexicon (diccionario) necesario para NLTK
nltk.download('vader_lexicon')

# Crear una instancia del analizador de sentimientos de NLTK
sia = SentimentIntensityAnalyzer()

# Título de la aplicación
st.title("Análisis de Sentimiento y Recomendación de Texto en Inglés")

# Entrada de texto del usuario para el análisis de sentimiento
texto_usuario = st.text_area("Ingresa el texto en inglés que deseas analizar:")

# Carga de archivo CSV
st.sidebar.header("Cargar datos desde CSV")
uploaded_file = st.sidebar.file_uploader("Cargar un archivo CSV", type=["csv"])

# Verificar si se ha cargado un archivo CSV
if uploaded_file is not None:
    # Leer el archivo CSV en un DataFrame de pandas
    df = pd.read_csv(uploaded_file)

    # Ver los primeros registros del DataFrame
    st.sidebar.subheader("Primeros registros del DataFrame:")
    st.sidebar.write(df.head())

    # Análisis de sentimiento en el DataFrame
    st.subheader("Análisis de Sentimiento en el DataFrame:")
    st.write(df)

    # Filtrar y eliminar filas con valores NaN en la columna "text"
    df = df.dropna(subset=['text'])

    # Calcular el análisis de sentimiento en la columna "text" del DataFrame (después de eliminar NaN)
    df['Sentimiento'] = df['text'].apply(lambda x: sia.polarity_scores(str(x))['compound'])

    # Recomendación basada en el análisis de sentimiento del texto del usuario
    if texto_usuario:
        # Calcular el análisis de sentimiento para el texto del usuario
        sentimiento_usuario = sia.polarity_scores(texto_usuario)

        # Determinar el sentimiento general del texto del usuario
        if sentimiento_usuario['compound'] >= 0.05:
            resultado_sentimiento_usuario = "Sentimiento positivo"
        elif sentimiento_usuario['compound'] <= -0.05:
            resultado_sentimiento_usuario = "Sentimiento negativo"
        else:
            resultado_sentimiento_usuario = "Sentimiento neutral"

        # Mostrar los resultados del análisis de sentimiento para el texto del usuario
        st.subheader("Análisis de Sentimiento para el Texto del Usuario:")
        st.write("Texto Ingresado:", texto_usuario)
        st.write("Resultado:", resultado_sentimiento_usuario)
        st.write("Puntaje de Sentimiento:", sentimiento_usuario['compound'])

        # Calcular la similitud de coseno entre el texto del usuario y los textos del DataFrame filtrados
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(df['text'])
        cosine_similarities_usuario = linear_kernel(tfidf_matrix, tfidf_vectorizer.transform([texto_usuario]))
        df['Similitud_Usuario'] = cosine_similarities_usuario

        # Ordenar por similitud y mostrar la recomendación con el "gmap_id"
        df_usuario = df.sort_values(by='Similitud_Usuario', ascending=False).head(5)
        st.subheader("Recomendación basada en el Texto del Usuario:")
        st.write(df_usuario[['gmap_id', 'name', 'description', 'Sentimiento', 'Similitud_Usuario']])

        # Mostrar los 5 primeros resultados en una tabla
        st.subheader("Los 5 primeros resultados de la recomendación:")
        st.write(df_usuario[['gmap_id', 'name', 'description', 'Sentimiento', 'Similitud_Usuario']].reset_index(drop=True))

        # Crear un mapa de Folium y agregar marcadores para las ubicaciones de los 5 primeros resultados
        st.subheader("Ubicaciones en el Mapa:")
        m = folium.Map(location=[df_usuario['latitude'].mean(), df_usuario['longitude'].mean()], zoom_start=10)

        for index, row in df_usuario.iterrows():
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=row['name'] + ": " + row['description']
            ).add_to(m)

        # Mostrar el mapa en Streamlit usando st.components.v1.html
        st.components.v1.html(m._repr_html_(), width=700, height=500)
