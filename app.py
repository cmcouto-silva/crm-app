import pickle
import pandas as pd
import streamlit as st

# Page config
st.set_page_config(page_title='CRM App')

# Page title
st.title('Customer Service Prediction')

# -- Parameters -- #

col1, col2 = st.columns(2)

with col1:
    days_since_last_SE_GI = st.number_input(label='days_since_last_SE_GI', value=-1, min_value=-1, max_value=5945)
    gw_g8 = st.number_input(label='gw_g8', value=-14.95, min_value=-14.95, max_value=478379.80)
    to_l5_l20 = st.number_input(label='to_l5_l20', value=0.20, min_value=0.00, max_value=1.00)
    ini_bon_g10 = st.number_input(label='ini_bon_g10', value=0., min_value=0.00, max_value=6082.38)
    succ_dep_g10 = st.number_input(label='succ_dep_g10', value=60.00, min_value=0.00, max_value=115289.53)
    SE_GI_total_70days = st.number_input(label='SE_GI_total_70days', value=0, min_value=0, max_value=11)
    

with col2:
    days_g2 = st.number_input(label='days_g2', value=2, min_value=0, max_value=7)
    gw_g10 = st.number_input(label='gw_g10', value=-24.64, min_value=-73920.08, max_value=108287.85)
    GOC_to_g9 = st.number_input(label='GOC_to_g9', value=0., min_value=0.00, max_value=2768675.00)
    turnover_last_20days = st.number_input(label='turnover_last_20days', value=1553.36, min_value=0.01, max_value=5412425.14)
    succ_dep_cnt_g9 = st.number_input(label='succ_dep_cnt_g9', value=2, min_value=0, max_value=325)
    SE_GI_max_datediff = st.number_input(label='SE_GI_max_datediff', value=-1, min_value=-1, max_value=5015)


# -- Model -- #

with open('models/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

def prediction():
    df_input = pd.DataFrame([dict(
    days_since_last_SE_GI = days_since_last_SE_GI,
    gw_g8 = gw_g8,
    to_l5_l20 = to_l5_l20,
    ini_bon_g10 = ini_bon_g10,
    succ_dep_g10 = succ_dep_g10,
    SE_GI_total_70days = SE_GI_total_70days,
    days_g2 = days_g2,
    gw_g10 = gw_g10,
    GOC_to_g9 = GOC_to_g9,
    turnover_last_20days = turnover_last_20days,
    succ_dep_cnt_g9 = succ_dep_cnt_g9,
    SE_GI_max_datediff = SE_GI_max_datediff
    )])
    score = model.predict_proba(df_input)[0,1]
    return score

if st.button('Predict'):
    y_score = prediction()
    y_pred = int(y_score > 0.5)
    if y_pred:
        st.error(f'{y_pred} ({y_score:.2%})')
    else:
        st.success(f'{y_pred} ({y_score:.2%})')
