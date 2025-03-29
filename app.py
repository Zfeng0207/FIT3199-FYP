import pickle
from pathlib import Path
import streamlit as st
import streamlit_authenticator as stauth

#user authentication
names=['Peter Parker','Rebecca Miller']
usernames=['pparker','rmiller']
file_path=Path(__file__).parent / 'hashed_pw.pkl'
with file_path.open('rb') as file:
    hashed_passwords=pickle.load(file)

authenticator=stauth.Authenticate(names,usernames,hashed_passwords,'sales_dashboard','abcdef',cookie_expiry_days=0)
name,authentication_status,username=authenticator.login('Login','main')

if authentication_status is False:
    st.error('username/password incorrect')
if authentication_status is None:
    st.warning('please enter')
if authentication_status:
    st.write('big fan man, big fan')
    authenticator.logout('Logout','sidebar')