import streamlit as st
import pandas as pd
import numpy as np
st.markdown("""
            # Its my new website
            ## its a sub section
            ### its a sub sub section""")

df = pd.DataFrame({
    'first column': list(range(1, 11)),
    'second column': np.arange(10, 101, 10)
})

line_count = st.slider('Select a line count', 1, 10, 3)

# and used to select the displayed lines
head_df = df.head(line_count)
head_df
number = st.number_input('Insert a number')

st.write('The current number is ', number)
