import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Welcome to Human Leukemia Cytomorphology Dataset classification")

st.write("""
This application is to classify blood cell images that belong to the Human Leukemia Cytomorphology Dataset available in Kaggle""")
st.markdown("""
- [Dataset from Kaggle](https://www.kaggle.com/datasets/gchan357/human-aml-cytomorphology-dataset)
""")
st.write("""Convolutional neural network approach was used by employing transfer learning approach.
         VGG16 base model was used to extract the image features followed by dense network to classsify the images.
         The dataset comprised of more than 80,000 images from 5 diffrent classes
""")
st.write("""- (1) normal patients (control)
- (2) APL with PML::RARA fusion (PML_RARA)
- (3) AML with NPM1 mutation (NPM1)
- (4) AML with CBFB::MYH11 fusion without NPM1 mutation (CBFB_MYH11)
- (5) AML with RUNX1::RUNX1T1 fusion (RUNX1_RUNX1T1)""")

#Dataset Summary
patient_count={'CBFB_MYH11': 37,
 'NPM1': 36,
 'PML_RARA': 24,
 'RUNX1_RUNX1T1': 32,
 'control': 60}
sample_count={'CBFB_MYH11': 17212,
 'NPM1': 17710,
 'PML_RARA': 11584,
 'RUNX1_RUNX1T1': 14403,
 'control': 20305}
df=pd.DataFrame([sample_count,patient_count]).T
df.columns=["Sample_Count", "Patient_Count"]
fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2,figsize=(10,5))
fig.suptitle("Dataset Summary", color="r", fontsize=20)
df["Sample_Count"].plot(kind="bar",ax=ax1, colormap="RdBu")
df["Patient_Count"].plot(kind="bar",ax=ax2, colormap='RdBu')
ax1.set_ylabel("No of images",fontsize=14, color='darkblue', fontweight='bold')
ax2.set_ylabel("No of patients",fontsize=14, color='darkblue', fontweight='bold')
ax1.tick_params(axis='x', labelsize=11, labelcolor='b', labelrotation=45)
ax2.tick_params(axis='x', labelsize=11, labelcolor='b', labelrotation=45)
ax1.tick_params(axis='y', labelsize=11, labelcolor='b')
ax2.tick_params(axis='y', labelsize=11, labelcolor='b')
plt.subplots_adjust(wspace=5.0)
plt.tight_layout()
st.pyplot(fig)

st.markdown("""### Further Information from the kaggle""")
st.write("""A total of 189 peripheral blood smears from the Munich Leukemia Laboratory (MLL) database from
         the years 2009 to 2020 were digitized. First, all blood smears werbe scanned with 10x magnification and an
         overview image was created. Using the Metasystems Metafer platform, cell detection was performed automatically using a segmentation
         threshold and logarithmic color transformation. Further analysis regarding the quality of the region within the blood smear was performed automatically.
         Per patient, 99-500 white blood cells were then scanned in 40x magnification via oil immersion microscopy in .TIF format, corresponding to 24,9μm x 24,9μm (144x144 pixels).
         For this, a CMOS Color Camera from MetaSystems with a resolution of 4096x3000px and a pixel size of 3,45μm x 3,45μm was used.
         Four pixels were binned into one, leading to a size of 6.9μm x 6.9μm, and a resolution of 6.9μm / 40 (1px = 0,1725μm).""")
