import time

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity


@st.cache()
def fetch_and_clean_data(file_path):
    df = pd.read_csv(file_path)

    # get all items
    items = set()
    for x in df.ingredients:
        for val in x.split(', '):
            items.add(val.lower().strip())
        # break
    # items = sorted(items)

    # create new dataframe
    new_df = pd.DataFrame(data=np.zeros((255, 367), dtype=int), columns=['name', 'ingredients'] + list(items))


    for i, d in df.iterrows():
        new_df.loc[i, ['name', 'ingredients']] = d[:2]

        for val in d[1].split(', '):
            item = val.lower().strip()
            new_df.loc[i, item] = 1

    return new_df


def embed_query(query, it):
    embedding = np.zeros((365,), dtype=int)
    
    for q in query:
        idx = np.where(q == it)
        embedding[idx] = 1

    return embedding

data = fetch_and_clean_data('food_250.csv')
st.title('Food From Fridge')

available_items = st.multiselect(
    label = 'Select items that are available with you',
    options = data.columns[2:],
)

submit = st.button('Submit')

if submit:
    st.header('You can try to make these recipes')
    with st.spinner('Searching for recipes'):
        # time.sleep(3)
        emb_qy = embed_query(available_items, data.columns[2:].values)
        sim = cosine_similarity(data.iloc[:, 2:].values.reshape(255, -1), emb_qy.reshape(1, -1)).ravel()

        idx_sorted = np.argsort(sim)[::-1]

        for val, idx in np.column_stack((sim[idx_sorted], idx_sorted)):
            if val > 0:
                st.info(f'**{data.iloc[int(idx), 0]}** ({data.iloc[int(idx), 1]})')
                # st.subheader(data.iloc[int(idx), 0])
                # st.write(f'Ingredients = {data.iloc[int(idx), 1]}')
