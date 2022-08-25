import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder
from sklearn.preprocessing import StandardScaler

from sklearn import set_config
from sklearn.model_selection import RandomizedSearchCV



from treeinterpreter import treeinterpreter as ti
import os

path = os.path.dirname(__file__)
logo_filename = path + '/images/logo.png'

#page configuration
st.set_page_config(page_title='Individual Resilience Prediction webapp', page_icon='🧗')

st.markdown("""
<style>
.appview-container .main .block-container{max-width: 800px;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

#this is the header
t1, t2 = st.columns((0.3,1)) 

t1.image(logo_filename, width = 150)
t2.header('Individual Resilience Prediction webapp')

t2.write('''
A continuación puedes contestar a los siguientes cuestionarios para conocer la predicción sobre tu resiliencia:
* **TILS:** Three-Item Loneliness Scale (TILS; Hughes, Waite, Hawkley & Cacioppo, 2004).
* **Substance score:** Increased consumption of food, alcohol, drugs and tobacco.
* **OFS:** Openness to the Future Scale (OFS; Botella et al., 2018).
* **BRS:** Brief Resilience Scale (BRS; Smith et al., 2008).
* **DAI:** Death Anxiety Inventory (DAI; Tomás-Sábado et al., 2005).
* **IUS:** Intolerance to Uncertainty Scale (IUS; Buhr & Dufas, 2002).
* **PHI:** Pemberton Happiness Index (PHI; Hervás and Vázquez, 2013).
* **PTS:** Paranoid Thoughts Scale (G-PTS; Green et al., 2008).
        ''')
st.write('---')

#TILS for LONELI
st.header('Three-Item Loneliness Scale')
tils_1 = st.radio('¿Con qué frecuencia sientes que te falta compañía?', ('Casi nunca', 'A veces', 'A menudo'))
st.markdown('####')
tils_2 = st.radio('¿Con qué frecuencia te sientes excluido?', ('Casi nunca', 'A veces', 'A menudo'))
st.markdown('####')
tils_3 = st.radio('¿Con qué frecuencia se siente aislado de los demás?', ('Casi nunca', 'A veces', 'A menudo'))

#recoding answers
tils_dict = {'Casi nunca': '1', 'A veces': '2', 'A menudo': '3'}
for key in tils_dict.keys():
    tils_1 = tils_1.replace(key, tils_dict[key])
    tils_2 = tils_2.replace(key, tils_dict[key])
    tils_3 = tils_3.replace(key, tils_dict[key])

#calculating final value for feature
LONELI = np.sum((int(tils_1), int(tils_2), int(tils_3)))

st.write('---')

#SUBSTANCE_SCORE
st.header('Substance score')
st.write('Durante las últimas 2 semanas, ¿Con qué frecuencia has tenido las siguientes experiencias?')
subst_1 = st.radio('Has comido más de lo normal:', ('Ningún día', 'Menos de la mitad de los días', 'Más de la mitad de los días', 'Casi todos los días'))
st.markdown('####')
subst_2 = st.radio('Has bebido alcohol más de lo normal:', ('Ningún día', 'Menos de la mitad de los días', 'Más de la mitad de los días', 'Casi todos los días'))
st.markdown('####')
subst_3 = st.radio('Has fumado más de lo normal:', ('Ningún día', 'Menos de la mitad de los días', 'Más de la mitad de los días', 'Casi todos los días'))
st.markdown('####')
subst_4 = st.radio('Has consumido más psicofármacos de los que consumes habitualmente:', ('Ningún día', 'Menos de la mitad de los días', 'Más de la mitad de los días', 'Casi todos los días'))
st.markdown('####')
subst_5 = st.radio('Has consumido más drogas de las que consume habitualmente:', ('Ningún día', 'Menos de la mitad de los días', 'Más de la mitad de los días', 'Casi todos los días'))

#recoding answers
subst_dict = {'Ningún día': '0', 'Menos de la mitad de los días': '1', 'Más de la mitad de los días': '2', 'Casi todos los días': '3'}
for key in subst_dict.keys():
    subst_1 = subst_1.replace(key, subst_dict[key])
    subst_2 = subst_2.replace(key, subst_dict[key])
    subst_3 = subst_3.replace(key, subst_dict[key])
    subst_4 = subst_4.replace(key, subst_dict[key])
    subst_5 = subst_5.replace(key, subst_dict[key])

#calculating final value for feature
SUBSTANCE_SCORE = np.mean((int(subst_1), int(subst_2), int(subst_3), int(subst_4), int(subst_5)))

st.write('---')

#OFS for OFS_total
st.header('Openness to Future Scale')
st.write('A continuación, encontrarás diferentes frases con las que te puedes sentir identificado en mayor o menor medida. Por favor, indica el grado de acuerdo o de desacuerdo que tienes con cada una de ellas.\
        Puedes arrastrar el selector:')

ofs_1 = st.select_slider('Cuando hago planes, estoy seguro de qué conseguiré llevarlos a cabo:',
                          options=['Totalmente en desacuerdo', 'Algo en desacuerdo', 'Ni de acuerdo ni en desacuerdo', 'Algo de acuerdo', 'Totalmente de acuerdo'])
st.markdown('####')
ofs_2 = st.select_slider('Suelo confiar en que las cosas saldrán bien:',
                          options=['Totalmente en desacuerdo', 'Algo en desacuerdo', 'Ni de acuerdo ni en desacuerdo', 'Algo de acuerdo', 'Totalmente de acuerdo'])
st.markdown('####')
ofs_3 = st.select_slider('Creo que tengo bastante control sobre el rumbo que toma mi vida:',
                          options=['Totalmente en desacuerdo', 'Algo en desacuerdo', 'Ni de acuerdo ni en desacuerdo', 'Algo de acuerdo', 'Totalmente de acuerdo'])
st.markdown('####')
ofs_4 = st.select_slider('Los desafíos y los retos hacia el futuro me resultan muy estimulantes:',
                          options=['Totalmente en desacuerdo', 'Algo en desacuerdo', 'Ni de acuerdo ni en desacuerdo', 'Algo de acuerdo', 'Totalmente de acuerdo'])
st.markdown('####')
ofs_5 = st.select_slider('Tengo un montón de planes e ilusiones:',
                          options=['Totalmente en desacuerdo', 'Algo en desacuerdo', 'Ni de acuerdo ni en desacuerdo', 'Algo de acuerdo', 'Totalmente de acuerdo'])
st.markdown('####')
ofs_6 = st.select_slider('A veces me asusto y siento que pierdo el control cuando pienso en lo que podrá depararme la vida:',
                          options=['Totalmente en desacuerdo', 'Algo en desacuerdo', 'Ni de acuerdo ni en desacuerdo', 'Algo de acuerdo', 'Totalmente de acuerdo'])
st.markdown('####')
ofs_7 = st.select_slider('Acepto con tranquilidad que en la vida me van a ocurrir cosas buenas y malas:',
                          options=['Totalmente en desacuerdo', 'Algo en desacuerdo', 'Ni de acuerdo ni en desacuerdo', 'Algo de acuerdo', 'Totalmente de acuerdo'])
st.markdown('####')
ofs_8 = st.select_slider('Sé que la vida me presentará obstáculos, pero confío en qué podré superarlos:',
                          options=['Totalmente en desacuerdo', 'Algo en desacuerdo', 'Ni de acuerdo ni en desacuerdo', 'Algo de acuerdo', 'Totalmente de acuerdo'])
st.markdown('####')
ofs_9 = st.select_slider('Estoy de acuerdo con la afirmación: cada día es un nuevo día:',
                          options=['Totalmente en desacuerdo', 'Algo en desacuerdo', 'Ni de acuerdo ni en desacuerdo', 'Algo de acuerdo', 'Totalmente de acuerdo'])
st.markdown('####')
ofs_10 = st.select_slider('Tengo esperanza por lo que pueda traer el futuro:',
                          options=['Totalmente en desacuerdo', 'Algo en desacuerdo', 'Ni de acuerdo ni en desacuerdo', 'Algo de acuerdo', 'Totalmente de acuerdo']) 

#recoding answers
ofs_dict = {'Totalmente en desacuerdo': '1', 'Algo en desacuerdo': '2', 'Ni de acuerdo ni en desacuerdo': '3', 'Algo de acuerdo': '4', 'Totalmente de acuerdo': '5'}
for key in ofs_dict.keys():
	ofs_1 = ofs_1.replace(key, ofs_dict[key])
	ofs_2 = ofs_2.replace(key, ofs_dict[key])
	ofs_3 = ofs_3.replace(key, ofs_dict[key])
	ofs_4 = ofs_4.replace(key, ofs_dict[key])
	ofs_5 = ofs_5.replace(key, ofs_dict[key])
	ofs_6 = ofs_6.replace(key, ofs_dict[key])
	ofs_7 = ofs_7.replace(key, ofs_dict[key])
	ofs_8 = ofs_8.replace(key, ofs_dict[key])
	ofs_9 = ofs_9.replace(key, ofs_dict[key])
	ofs_10 = ofs_10.replace(key, ofs_dict[key])

#calculating final value for feature
OFS_total = np.sum((int(ofs_1), int(ofs_2), int(ofs_3), int(ofs_4), int(ofs_5), int(ofs_6), int(ofs_7), int(ofs_8), int(ofs_9), int(ofs_10)))

st.write('---')

#BRS for BRS_total
st.header('Brief Resilience Scale')
st.write('Utiliza la siguiente escala e indica para cada afirmación cuánto estás en desacuerdo o de acuerdo con cada una de las afirmaciones. Puedes arrastrar el selector:')

brs_1 = st.select_slider('Tiendo a recuperarme rápidamente después de los tiempos difíciles:',
                          options=['Totalmente en desacuerdo', 'En desacuerdo', 'Neutral', 'De acuerdo', 'Totalmente de acuerdo'])
st.markdown('####')
brs_2 = st.select_slider('Me resulta difícil superar los eventos estresantes:',
                          options=['Totalmente en desacuerdo', 'En desacuerdo', 'Neutral', 'De acuerdo', 'Totalmente de acuerdo'])
st.markdown('####')
brs_3 = st.select_slider('No me lleva mucho tiempo recuperarme de un evento estresante:',
                          options=['Totalmente en desacuerdo', 'En desacuerdo', 'Neutral', 'De acuerdo', 'Totalmente de acuerdo'])
st.markdown('####')
brs_4 = st.select_slider('Es difícil para mí recuperarme cuando algo malo sucede:',
                          options=['Totalmente en desacuerdo', 'En desacuerdo', 'Neutral', 'De acuerdo', 'Totalmente de acuerdo'])
st.markdown('####')
brs_5 = st.select_slider('Normalmente supero momentos complicados sin dificultad:',
                          options=['Totalmente en desacuerdo', 'En desacuerdo', 'Neutral', 'De acuerdo', 'Totalmente de acuerdo'])
st.markdown('####')
brs_6 = st.select_slider('Me lleva mucho tiempo superar los contratiempos en mi vida:',
                          options=['Totalmente en desacuerdo', 'En desacuerdo', 'Neutral', 'De acuerdo', 'Totalmente de acuerdo'])
st.markdown('####')

#recoding answers
brs_pos_dict = {'Totalmente en desacuerdo': '1', 'En desacuerdo': '2', 'Neutral': '3', 'De acuerdo': '4', 'Totalmente de acuerdo': '5'}
for key in brs_pos_dict.keys():
	brs_1 = brs_1.replace(key, brs_pos_dict[key])
	brs_3 = brs_3.replace(key, brs_pos_dict[key])
	brs_5 = brs_5.replace(key, brs_pos_dict[key])
	
brs_neg_dict = {'Totalmente en desacuerdo': '5', 'En desacuerdo': '4', 'Neutral': '3', 'De acuerdo': '2', 'Totalmente de acuerdo': '1'}
for key in brs_neg_dict.keys():
	brs_2 = brs_2.replace(key, brs_neg_dict[key])
	brs_4 = brs_4.replace(key, brs_neg_dict[key])
	brs_6 = brs_6.replace(key, brs_neg_dict[key])

#calculating final value for feature
BRS_total = np.mean((int(brs_1), int(brs_2), int(brs_3), int(brs_4), int(brs_5), int(brs_6)))

#DAI for DAI_TOTAL
st.header('Death Anxiety Inventory')
st.write('Por favor, indica en qué medida estás de acuerdo con las siguientes afirmaciones. Puedes arrastrar el selector:')

dai_1 = st.select_slider('La certeza de la muerte quita significado a la vida:',
                          options=['Totalmente en desacuerdo', 'En desacuerdo', 'Ni de acuerdo ni en desacuerdo', 'De acuerdo', 'Totalmente de acuerdo'])
st.markdown('####')
dai_2 = st.select_slider('Creo que tengo más miedo a la muerte que la mayoría de las personas:',
                          options=['Totalmente en desacuerdo', 'En desacuerdo', 'Ni de acuerdo ni en desacuerdo', 'De acuerdo', 'Totalmente de acuerdo'])
st.markdown('####')
dai_3 = st.select_slider('Los ataúdes me ponen nervioso:',
                          options=['Totalmente en desacuerdo', 'En desacuerdo', 'Ni de acuerdo ni en desacuerdo', 'De acuerdo', 'Totalmente de acuerdo'])
st.markdown('####')
dai_4 = st.select_slider('Me preocupa lo que haya después de la muerte:',
                          options=['Totalmente en desacuerdo', 'En desacuerdo', 'Ni de acuerdo ni en desacuerdo', 'De acuerdo', 'Totalmente de acuerdo'])
st.markdown('####')
dai_5 = st.select_slider('Frecuentemente pienso en mi propia muerte:',
                          options=['Totalmente en desacuerdo', 'En desacuerdo', 'Ni de acuerdo ni en desacuerdo', 'De acuerdo', 'Totalmente de acuerdo'])
st.markdown('####')

#recoding answers
dai_dict = {'Totalmente en desacuerdo': '1', 'En desacuerdo': '2', 'Ni de acuerdo ni en desacuerdo': '3', 'De acuerdo': '4', 'Totalmente de acuerdo': '5'}
for key in dai_dict.keys():
	dai_1 = dai_1.replace(key, dai_dict[key])
	dai_2 = dai_2.replace(key, dai_dict[key])
	dai_3 = dai_3.replace(key, dai_dict[key])
	dai_4 = dai_4.replace(key, dai_dict[key])
	dai_5 = dai_5.replace(key, dai_dict[key])

#calculating final value for feature
DAI_TOTAL = np.sum((int(dai_1), int(dai_2), int(dai_3), int(dai_4), int(dai_5)))

st.write('---')

#IUS for IUS_total
st.header('Intolerance to Uncertainty Scale')
st.write('A continuación encontrarás una serie de afirmaciones que describen cómo pueden reaccionar las personas ante las incertidumbres de la vida.\
	Por favor, utiliza la siguiente escala para describir hasta qué punto cada elemento es característico de ti. Puedes arrastrar el selector:')

ius_1 = st.select_slider('Los imprevistos me molestan mucho:',
                          options=['1. Nada característico en mí', '2', '3. Algo característico de mí', '4', '5. Totalmente característico de mí'])
st.markdown('####')
ius_2 = st.select_slider('Es frustrante para mí no tener toda la información que necesito:',
                          options=['1. Nada característico en mí', '2', '3. Algo característico de mí', '4', '5. Totalmente característico de mí'])
st.markdown('####')
ius_3 = st.select_slider('Se debería prever todo para evitar las sorpresas:',
                          options=['1. Nada característico en mí', '2', '3. Algo característico de mí', '4', '5. Totalmente característico de mí'])
st.markdown('####')
ius_4 = st.select_slider('Un pequeño imprevisto puede arruinarlo todo, incluso con la mejor de las planificaciones:',
                          options=['1. Nada característico en mí', '2', '3. Algo característico de mí', '4', '5. Totalmente característico de mí'])
st.markdown('####')
ius_5 = st.select_slider('Quiero saber siempre qué me depara el futuro:',
                          options=['1. Nada característico en mí', '2', '3. Algo característico de mí', '4', '5. Totalmente característico de mí'])
st.markdown('####')
ius_6 = st.select_slider('No soporto que me cojan por sorpresa:',
                          options=['1. Nada característico en mí', '2', '3. Algo característico de mí', '4', '5. Totalmente característico de mí'])
st.markdown('####')
ius_7 = st.select_slider('Tendría que ser capaz de organizar todo de antemano:',
                          options=['1. Nada característico en mí', '2', '3. Algo característico de mí', '4', '5. Totalmente característico de mí'])
st.markdown('####')
ius_8 = st.select_slider('La incertidumbre me impide disfrutar plenamente de la vida:',
                          options=['1. Nada característico en mí', '2', '3. Algo característico de mí', '4', '5. Totalmente característico de mí'])
st.markdown('####')
ius_9 = st.select_slider('Cuando llega el momento de actuar, la incertidumbre me paraliza:',
                          options=['1. Nada característico en mí', '2', '3. Algo característico de mí', '4', '5. Totalmente característico de mí'])
st.markdown('####')
ius_10 = st.select_slider('Cuando estoy indeciso no puedo funcionar muy bien:',
                          options=['1. Nada característico en mí', '2', '3. Algo característico de mí', '4', '5. Totalmente característico de mí'])
st.markdown('####')
ius_11 = st.select_slider('La más mínima duda me puede impedir actuar:',
                          options=['1. Nada característico en mí', '2', '3. Algo característico de mí', '4', '5. Totalmente característico de mí'])
st.markdown('####')
ius_12 = st.select_slider('Debo alejarme de toda situación incierta:',
                          options=['1. Nada característico en mí', '2', '3. Algo característico de mí', '4', '5. Totalmente característico de mí'])
st.markdown('####')

#recoding answers
ius_dict = {'1. Nada característico en mí': '1', '3. Algo característico de mí': '3', '5. Totalmente característico de mí': '5'}
for key in ius_dict.keys():
	ius_1 = ius_1.replace(key, ius_dict[key])
	ius_2 = ius_2.replace(key, ius_dict[key])
	ius_3 = ius_3.replace(key, ius_dict[key])
	ius_4 = ius_4.replace(key, ius_dict[key])
	ius_5 = ius_5.replace(key, ius_dict[key])
	ius_6 = ius_6.replace(key, ius_dict[key])
	ius_7 = ius_7.replace(key, ius_dict[key])
	ius_8 = ius_8.replace(key, ius_dict[key])
	ius_9 = ius_9.replace(key, ius_dict[key])
	ius_10 = ius_10.replace(key, ius_dict[key])
	ius_11 = ius_11.replace(key, ius_dict[key])
	ius_12 = ius_12.replace(key, ius_dict[key])

#calculating final value for feature
IUS_total = np.sum((int(ius_1), int(ius_2), int(ius_3), int(ius_4), int(ius_5), int(ius_6), int(ius_7), int(ius_8), int(ius_9), int(ius_10), int(ius_11), int(ius_12)))

st.write('---')

#PHI for PEMBERTON_TOTAL
st.header('Pemberton Happiness Index')
st.write('Por favor, usando la siguiente escala del 0 al 10, donde 0 significa totalmente en desacuerdo y 10 totalmente de acuerdo, elige en qué medida estás de acuerdo con las siguientes afirmaciones. \
	Puedes arrastrar el selector:')

phi_1 = st.select_slider('Me siento muy satisfecho con mi vida:',
						options=['0. Totalmente en desacuerdo', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10. Totalmente de acuerdo'],
						value = '5')
st.markdown('####')
phi_2 = st.select_slider('Me siento con la energía necesaria para cumplir bien mis tareas cotidianas:',
						options=['0. Totalmente en desacuerdo', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10. Totalmente de acuerdo'],
						value = '5')
st.markdown('####')
phi_3 = st.select_slider('Siento que mi vida es útil y valiosa:',
						options=['0. Totalmente en desacuerdo', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10. Totalmente de acuerdo'],
						value = '5')
st.markdown('####')
phi_4 = st.select_slider('Me siento satisfecho con mi forma de ser:',
						options=['0. Totalmente en desacuerdo', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10. Totalmente de acuerdo'],
						value = '5')
st.markdown('####')
phi_5 = st.select_slider('Mi vida está llena de aprendizajes y desafíos que me hacen crecer:',
						options=['0. Totalmente en desacuerdo', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10. Totalmente de acuerdo'],
						value = '5')
st.markdown('####')
phi_6 = st.select_slider('Me siento muy unido a las personas que me rodean:',
						options=['0. Totalmente en desacuerdo', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10. Totalmente de acuerdo'],
						value = '5')
st.markdown('####')
phi_7 = st.select_slider('Me siento capaz de resolver la mayoría de los problemas de mi día a día:',
						options=['0. Totalmente en desacuerdo', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10. Totalmente de acuerdo'],
						value = '5')
st.markdown('####')
phi_8 = st.select_slider('Siento que en lo importante puedo ser yo mismo:',
						options=['0. Totalmente en desacuerdo', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10. Totalmente de acuerdo'],
						value = '5')
st.markdown('####')
phi_9 = st.select_slider('Disfruto cada día de muchas pequeñas cosas:',
						options=['0. Totalmente en desacuerdo', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10. Totalmente de acuerdo'],
						value = '5')
st.markdown('####')
phi_10 = st.select_slider('En mi día a día tengo muchos ratos en los que me siento mal:',
						options=['0. Totalmente en desacuerdo', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10. Totalmente de acuerdo'],
						value = '5')
st.markdown('####')
phi_11 = st.select_slider('Siento que vivo en una sociedad que me permite desarrollarme plenamente:',
						options=['0. Totalmente en desacuerdo', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10. Totalmente de acuerdo'],
						value = '5')
st.markdown('####')

st.write('Por favor, marca ahora cuál de las siguientes cosas te sucedió ayer:')

phi_exp_1 = st.radio('Me sentí satisfecho por algo que hice:', ('Sí', 'No'))
st.markdown('####')
phi_exp_2 = st.radio('En algunos momentos me sentí desbordado:', ('Sí', 'No'))
st.markdown('####')
phi_exp_3 = st.radio('Pasé un rato divertido con alguien:', ('Sí', 'No'))
st.markdown('####')
phi_exp_4 = st.radio('Me aburrí durante bastante tiempo:', ('Sí', 'No'))
st.markdown('####')
phi_exp_5 = st.radio('Hice algo que realmente disfruto haciendo:', ('Sí', 'No'))
st.markdown('####')
phi_exp_6 = st.radio('Estuve preocupado por temas personales:', ('Sí', 'No'))
st.markdown('####')
phi_exp_7 = st.radio('Aprendí algo interesante:', ('Sí', 'No'))
st.markdown('####')
phi_exp_8 = st.radio('Pasaron cosas que me enfadaron mucho:', ('Sí', 'No'))
st.markdown('####')
phi_exp_9 = st.radio('Me permití un capricho:', ('Sí', 'No'))
st.markdown('####')
phi_exp_10 = st.radio('Me sentí menospreciado por alguien:', ('Sí', 'No'))
st.markdown('####')

#recoding answers
phi_dict = {'0. Totalmente en desacuerdo': '0', '10. Totalmente de acuerdo': '10'}
for key in phi_dict.keys():
	phi_1 = phi_1.replace(key, phi_dict[key])
	phi_2 = phi_2.replace(key, phi_dict[key])
	phi_3 = phi_3.replace(key, phi_dict[key])
	phi_4 = phi_4.replace(key, phi_dict[key])
	phi_5 = phi_5.replace(key, phi_dict[key])
	phi_6 = phi_6.replace(key, phi_dict[key])
	phi_7 = phi_7.replace(key, phi_dict[key])
	phi_8 = phi_8.replace(key, phi_dict[key])
	phi_9 = phi_9.replace(key, phi_dict[key])
	phi_11 = phi_11.replace(key, phi_dict[key])

phi_10_dict = {'0. Totalmente en desacuerdo': '10', '1': '9', '2': '8', '3': '7', '4': '6', '6': '4', '7': '3', '8': '2', '9': '1', '10. Totalmente de acuerdo': '0'}
for key in phi_10_dict.keys():
	phi_10 = phi_10.replace(key, phi_10_dict[key])

phi_pos_exp_dict = {'Sí': '1', 'No': '0'}
for key in phi_pos_exp_dict.keys():
	phi_exp_1 = phi_exp_1.replace(key, phi_pos_exp_dict[key])
	phi_exp_3 = phi_exp_3.replace(key, phi_pos_exp_dict[key])
	phi_exp_5 = phi_exp_5.replace(key, phi_pos_exp_dict[key])
	phi_exp_7 = phi_exp_7.replace(key, phi_pos_exp_dict[key])
	phi_exp_9 = phi_exp_9.replace(key, phi_pos_exp_dict[key])

phi_neg_exp_dict = {'Sí': '0', 'No': '1'}
for key in phi_neg_exp_dict.keys():
	phi_exp_2 = phi_exp_2.replace(key, phi_neg_exp_dict[key])
	phi_exp_4 = phi_exp_4.replace(key, phi_neg_exp_dict[key])
	phi_exp_6 = phi_exp_6.replace(key, phi_neg_exp_dict[key])
	phi_exp_8 = phi_exp_8.replace(key, phi_neg_exp_dict[key])
	phi_exp_10 = phi_exp_10.replace(key, phi_neg_exp_dict[key])

#calculating final value for feature
PEMBERTON_EXP = np.sum((int(phi_exp_1), int(phi_exp_2), int(phi_exp_3), int(phi_exp_4), int(phi_exp_5), int(phi_exp_6), int(phi_exp_7), int(phi_exp_8), int(phi_exp_9), int(phi_exp_10)))
PEMBERTON_TOTAL = np.mean((int(phi_1), int(phi_2), int(phi_3), int(phi_4), int(phi_5), int(phi_6), int(phi_7), int(phi_8), int(phi_9), int(phi_10), int(phi_11), int(PEMBERTON_EXP)))

st.write('---')

#PTS for PADS_TOTAL
st.header('Paranoid Thoughts Scale')
st.write('Por favor, indica hasta qué punto estás de acuerdo con las siguientes afirmaciones. Puedes arrastrar el selector:')

pts_1 = st.select_slider('Mis amigos suelen decirme que deje de preocuparme por si me engañan o me hacen daño:',
                          options=['Completamente en desacuerdo', 'En desacuerdo', 'Ni de acuerdo ni en desacuerdo', 'De acuerdo', 'Completamente de acuerdo'])
st.markdown('####')
pts_2 = st.select_slider('A menudo sospecho de las intenciones de los demás hacia mí:',
                          options=['Completamente en desacuerdo', 'En desacuerdo', 'Ni de acuerdo ni en desacuerdo', 'De acuerdo', 'Completamente de acuerdo'])
st.markdown('####')
pts_3 = st.select_slider('Es casi seguro que la gente me mentirá:',
                          options=['Completamente en desacuerdo', 'En desacuerdo', 'Ni de acuerdo ni en desacuerdo', 'De acuerdo', 'Completamente de acuerdo'])
st.markdown('####')
pts_4 = st.select_slider('Creo que algunas personas quieren hacerme daño deliberadamente:',
                          options=['Completamente en desacuerdo', 'En desacuerdo', 'Ni de acuerdo ni en desacuerdo', 'De acuerdo', 'Completamente de acuerdo'])
st.markdown('####')
pts_5 = st.select_slider('Sólo debes confiar en ti mismo:',
                          options=['Completamente en desacuerdo', 'En desacuerdo', 'Ni de acuerdo ni en desacuerdo', 'De acuerdo', 'Completamente de acuerdo'])
st.markdown('####')

#recoding answers
pts_dict = {'Completamente en desacuerdo': '0', 'En desacuerdo': '1', 'Ni de acuerdo ni en desacuerdo': '2', 'De acuerdo': '3', 'Completamente de acuerdo': '4'}
for key in pts_dict.keys():
	pts_1 = pts_1.replace(key, pts_dict[key])
	pts_2 = pts_2.replace(key, pts_dict[key])
	pts_3 = pts_3.replace(key, pts_dict[key])
	pts_4 = pts_4.replace(key, pts_dict[key])
	pts_5 = pts_5.replace(key, pts_dict[key])

#calculating final value for feature
PADS_TOTAL = np.sum((int(pts_1), int(pts_2), int(pts_3), int(pts_4), int(pts_5)))

st.write('---')

instance = pd.DataFrame(data = {'SUBSTANCE_SCORE': [SUBSTANCE_SCORE],
								'PADS_TOTAL': [PADS_TOTAL],
								'IUS_total': [IUS_total],
								'PEMBERTON_TOTAL': [PEMBERTON_TOTAL],
								'LONELI': [LONELI],
								'DAI_TOTAL': [DAI_TOTAL],
								'BRS_total': [BRS_total],
								'OFS_total': [OFS_total]
								}, 
								dtype = np.float16)

st.write(instance)

#load the model from disk
model_filename = path + '/RandomForestClassifier_NoCovidFeatures_Webapp_model.sav'
loaded_model = pickle.load(open(model_filename, 'rb'))

#make prediction
prediction_class = loaded_model.predict(instance)
prediction_proba = loaded_model.predict_proba(instance)

#find feature contributions
prediction_ti, bias, contributions = ti.predict(loaded_model, instance)
st.write('Prediction', prediction_ti)
st.write('Bias (trainset prior)', bias)
st.write('Feature contributions:')
for c, feature in zip(contributions[0], loaded_model.feature_names_in_):
	st.write(feature, c)

if prediction_class == 0:
	st.write('No eres resiliente, con un score de predicción igual a', round(prediction_proba[0, 0], 2))
elif prediction_class == 1:
	st.write('Eres resiliente, con un score de predicción igual a', round(prediction_proba[0, 1], 2))

#def predecir_resiliencia(instance_vector):


#st.button('Predecir mi resiliencia', )