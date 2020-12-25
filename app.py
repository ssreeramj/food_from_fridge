import pandas as pd
import numpy as np
import streamlit as st


@st.cache()
def fetch_and_clean_data(file_path):
    df = pd.read_csv(file_path)

    # get all items
    items = set()
    for x in df.ingredients:
        for val in x.split(', '):
            items.add(val.lower().strip())
        # break

    # create new dataframe
    new_df = pd.DataFrame(data=np.zeros((255, 366), dtype=int), columns=['name'] + list(items))

    for i, d in df.iterrows():
        new_df.loc[i, 'name'] = d[0]
    
        for val in d[1].split(', '):
            item = val.lower().strip()
            new_df.loc[i, item] = 1

    return new_df

data = fetch_and_clean_data('food_250.csv')

st.title('Food From Fridge')

available_items = st.multiselect(
    label = 'Select items that are available with you',
    options = data.columns[1:],
)


finalize_items = st.button('Submit')



