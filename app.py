import pickle
from pathlib import Path
import joblib
import streamlit as st
import streamlit_authenticator as stauth
import pandas as pd

st.title('Authentication')
if st.button('Authenticate'):
    st.login('google')