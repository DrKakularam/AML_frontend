import streamlit as st
import requests
import numpy as np
st.title("Image Classification API Demo")
st.markdown("""### Try blood cell image from the dataset to find the prediction""")
class_to_index={'CBFB_MYH11': 0, 'NPM1': 1, 'PML_RARA': 2, 'RUNX1_RUNX1T1': 3, 'control': 4}
index_to_class={0: 'CBFB_MYH11', 1: 'NPM1', 2: 'PML_RARA', 3: 'RUNX1_RUNX1T1', 4: 'control'}
aml_class = st.radio('Select a class', (class_to_index.keys()))
patient_groups = {
    'CBFB_MYH11': [
        'XMA', 'NXR', 'FKU', 'BTB', 'MSC', 'OUR', 'IEN', 'OEY', 'FOL', 'SBY',
        'DSN', 'GOP', 'VDQ', 'OHZ', 'YXH', 'BSN', 'FPW', 'XBB', 'ZRJ', 'DQV',
        'ZEE', 'XIE', 'HQE', 'BJK', 'NCD', 'KOV', 'AQK', 'PCQ', 'ZVS', 'GOR',
        'FYX', 'ECB', 'POM', 'QCF', 'HFM', 'MFH', 'ZNB'
    ],
    'NPM1': [
        'XFF', 'TPI', 'YCY', 'SYY', 'YPT', 'BHR', 'UXN', 'NPA', 'ZEQ', 'WTE',
        'IQS', 'OCV', 'RPA', 'SDP', 'NHB', 'ZMX', 'QQD', 'RFV', 'PAM', 'ZLQ',
        'QBU', 'UVT', 'ZLW', 'DPF', 'LGB', 'FOH', 'JTF', 'HXQ', 'ALA', 'CVW',
        'PBO', 'YRZ', 'SQF', 'LBW', 'CZI', 'EHM'
    ],
    'PML_RARA': [
        'BIK', 'JFC', 'NUS', 'RNQ', 'OYO', 'TXH', 'ALE', 'JRB', 'GOJ', 'GQB',
        'YST', 'CMR', 'BHS', 'UZP', 'LYD', 'GEG', 'YPF', 'ZZM', 'CWF', 'FTW',
        'PKC', 'VAJ', 'EEN', 'SUN'
    ],
    'RUNX1_RUNX1T1': [
        'ORE', 'FED', 'DHA', 'BHG', 'ZJJ', 'RHX', 'XXX', 'IIG', 'LPA', 'BKR',
        'BOE', 'RJQ', 'ABF', 'HEJ', 'GUE', 'UGU', 'KRG', 'ISW', 'XOB', 'ONF',
        'RGG', 'NRB', 'VXU', 'ZVX', 'FQY', 'HVE', 'SWN', 'UWF', 'XTK', 'ILZ',
        'HMC', 'VMO'
    ],
    'control': [
        'GJZ', 'ENH', 'IPY', 'NFI', 'NAR', 'MKF', 'ORD', 'LCW', 'GXQ', 'MOR',
        'MPP', 'YWQ', 'CDA', 'DJJ', 'AIH', 'CIQ', 'UOU', 'JGE', 'FDW', 'CCO',
        'YDL', 'NXO', 'RDV', 'ZNK', 'HNB', 'HQQ', 'YXY', 'NOL', 'LAM', 'QIZ',
        'DPU', 'TVZ', 'FAQ', 'AEC', 'NVD', 'NRL', 'DFT', 'WOA', 'OTU', 'WSD',
        'GJQ', 'UNX', 'BXR', 'DNX', 'UBG', 'OVP', 'MJF', 'DHE', 'XXJ', 'KDW',
        'VPN', 'TOK', 'TBZ', 'XXI', 'ATM', 'AVL', 'ZNL', 'WXO', 'ICS', 'FUT'
    ]
}
patient_id=st.selectbox("Select a Patient",(patient_groups[aml_class]))
image_num = st.slider('Select an image', 0,500)

api_url = "http://localhost:8000/image_dataset/"
image_path= f"aml_data/{aml_class}/{patient_id}_image_{image_num}.tif"
response=requests.get(api_url, params={"image_path":image_path})
prediction=response.json()["prediction"][0]
if len(prediction)==5:
    index=np.argmax(response.json()["prediction"])
    st.markdown("### Prediction Results")
    st.write({"Prediction class": index_to_class[index], "Actual class":aml_class})
    st.write({"Prediction probability": prediction[index]})
else:
    st.markdown("### Image not Found")
