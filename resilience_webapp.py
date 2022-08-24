import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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

t1.image('images/logo.png', width = 250)
t2.header('Individual Resilience Prediction webapp')

t2.write('''
A continuación puedes contestar a los siguientes cuestionarios para conocer la predicción sobre tu resiliencia:
* **Cuestionario 1:** es un cuestionario.
* **Cuestionario 2:** es otro cuestionario.
        ''')
st.write('---')