import numpy as np
import pandas as pd


def prepare_mimic_ecg(digits, target_folder, df_mapped=None, df_diags=None):

    print("preparing MIMIC ECG dataset for finetuning")
    
    # load label dataframe
    if df_diags is not None:
        df_diags = df_diags
    else:
        if((target_folder/"records_w_diag_icd10.pkl").exists()):
            df_diags = pd.read_pickle(target_folder/"records_w_diag_icd10.pkl")
        else:
            df_diags = pd.read_csv(target_folder/"records_w_diag_icd10.csv")
            # df_diags.drop('Unnamed: 0',axis=1, inplace=True)
            df_diags['ecg_time']=pd.to_datetime(df_diags["ecg_time"])
            df_diags['dod']=pd.to_datetime(df_diags["dod"])
            for c in ['ed_diag_ed', 'ed_diag_hosp', 'hosp_diag_hosp', 'all_diag_hosp', 'all_diag_all']:
                df_diags[c]=df_diags[c].apply(lambda x: eval(x))

    df_diags["label_train"]=df_diags["all_diag_all"]
    labelsettrain=labelsettrain[len("all"):]

    df_diags["has_statements_train"]=df_diags["label_train"].apply(lambda x: len(x)>0)#keep track if there were any ICD statements for this sample
    

    #first truncate to desired number of digits
    if(digits is not None):
        df_diags["label_train"]=df_diags["label_train"].apply(lambda x: list(set([y.strip()[:digits] for y in x])))
    
    #remove trailing placeholder Xs    
    df_diags["label_train"]=df_diags["label_train"].apply(lambda x: list(set([y.rstrip("X") for y in x])))

    print("filter stroke")
    #no selection = apply, specific icd codes that are linked to stroke
    df_diags["label_stroke"] = df_diags["label_train"].apply(
        lambda x: int(any(c.startswith(('I60', 'I61', 'I62', 'I63', 'I64', 'I65')) for c in x)))
        
    return df_diags