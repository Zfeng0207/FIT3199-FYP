import pickle as pkl
import pandas as pd
with open("df_memmap.pkl", "rb") as f:
    object = pkl.load(f)
    
df = pd.DataFrame(object)
df.to_csv(r'df_memmap.csv')