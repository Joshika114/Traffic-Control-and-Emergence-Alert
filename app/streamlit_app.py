import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.preprocessing import load_dataset, clean_crash_data

st.set_page_config(layout="wide")
st.title("🚦 Smart Traffic Accident Prediction and Analysis")

df = clean_crash_data(load_dataset("data/Traffic_Crashes_-_Crashes.csv"))

st.header("Top 5 Records")
st.dataframe(df.head())

st.subheader("Weather Conditions")
st.bar_chart(df['WEATHER_CONDITION'].value_counts())

st.subheader("Correlation Heatmap")
corr = df.select_dtypes(include='number').corr()
fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, ax=ax, cmap='coolwarm')
st.pyplot(fig)
