import streamlit as st
import requests
import numpy as np
st.title("Image Classification API Demo")
st.markdown("""## Upload blood cell image from local computer to find the prediction""")

# Image upload UI
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp", "tiff"])
index_to_class={0: 'CBFB_MYH11', 1: 'NPM1', 2: 'PML_RARA', 3: 'RUNX1_RUNX1T1', 4: 'control'}
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", width=300)
    st.write("")

    if st.button("Classify Image"):
        # Prepare files payload for API
        files = {"image": uploaded_file.getvalue()}
        # Replace the URL below with your actual API endpoint
        api_url = "http://localhost:8000/upload_image/"

        # Make the POST request
        response = requests.post(api_url, files={"image": (uploaded_file.name, uploaded_file, uploaded_file.type)})

        if response.status_code == 200:
            prediction=response.json()["prediction"][0]
            index=np.argmax(prediction)
            st.markdown("### Prediction Results")
            st.write({"Prediction class": index_to_class[index],
                        "Prediction probability": prediction[index]})
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
