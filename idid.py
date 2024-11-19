import pandas as pd
name = '3D printing'
PATH = f'data/graphs/{name}_papers.csv' 
data = pd.read_csv(PATH)
title = 'Hydrocolloids to Control the Texture of Three‐Dimensional (3D)‐Printed Foods'
filtered_data = data[data['title'] == title]
id = filtered_data['id'].tolist()
print(id)