import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle
from sklearn.ensemble import RandomForestClassifier
import os

path = os.path.dirname(__file__)
logo_file = path + '/images/logo.png'

#page configuration
st.set_page_config(page_title='Individual Resilience Prediction webapp', page_icon='🧗')

st.markdown("""
<style>
.appview-container .main .block-container{max-width: 1200px;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

#this is the header
t1, t2 = st.columns((0.3,1)) 

t1.image(logo_file, width = 250)
t2.header('Individual Resilience Prediction webapp')

t2.write('''
A continuación puedes contestar a los siguientes cuestionarios para conocer la predicción sobre tu resiliencia:
* **TILS:** Three-Item Loneliness Scale (TILS; Hughes, Waite, Hawkley & Cacioppo, 2004).
* **Cuestionario 2:** es otro cuestionario.
        ''')
st.write('---')

#TILS para calcular LONELI
st.header('Three-Item Loneliness Scale')
tils_1 = st.radio('¿Con qué frecuencia sientes que te falta compañía?', ('Casi nunca', 'A veces', 'A menudo'))
print(tils_1)
tils_2 = st.radio('¿Con qué frecuencia te sientes excluido?', ('Casi nunca', 'A veces', 'A menudo'))
print(tils_2)
tils_3 = st.radio('¿Con qué frecuencia se siente aislado de los demás?', ('Casi nunca', 'A veces', 'A menudo'))
print(tils_3)