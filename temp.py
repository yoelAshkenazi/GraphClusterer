import pandas
import pickle as pkl

import pandas as pd

data = pd.read_csv('data/posts_content_parsed.csv')

# make a smaller version with 5000 rows selected randomly
data_sample = data.sample(5000)
data_sample['id'] = range(1, 5001)

# save to csv.
data_sample.to_csv('data/posts_5k_sampled.csv', index=False)
