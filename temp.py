import pandas
import pickle as pkl

import pandas as pd

data = pd.read_parquet('data/posts_content_parsed.parquet')

# Make an 'id' column starting at 1.
data['id'] = range(1, len(data) + 1)
# Change the column name from 'content' to 'abstract'.
data = data.rename(columns={'content': 'abstract'})

# Save the data to a csv file.
data.to_csv('data/posts_content_parsed.csv', index=False)
