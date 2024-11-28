import pandas as pd

name = '3D printing'
PATH = f'data/graphs/{name}_papers.csv'
data = pd.read_csv(PATH)[['id', 'abstract']]

url = 'https://openalex.org/W3032846455'

# Filter and print the abstract
abstract = data.loc[data['id'] == peripheral_url, 'abstract'].iloc[0]
print(abstract)
