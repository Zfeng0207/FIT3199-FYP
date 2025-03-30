# FIT3199-FYP
Libraries to download = streamlit, streamlit-authenticator
For streamlit-authenticator make sure it is version 0.1.5
Streamlit authenticator install command: pip install streamlit-authenticator==0.1.5 

username='pparker'
password='abc123'
To create new username and passwords:
1. Go to generate_keys.py
2. Add username and password to the usernames and passwords list respectively
3. Run generate_keys.py
4. Username and passwords will be stored in hashed_pw.pkl. Do not delete it.

To run open vscode then type streamlit app.py in terminal
Common problem:
1. Problem: model not defined
Fix: change the model path to the absolute path in your local file
