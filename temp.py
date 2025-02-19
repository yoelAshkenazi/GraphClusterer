import pandas
import pickle as pkl

import pandas as pd


# data = pd.read_csv('data/newsgroups.csv')
#
# # Rename the content column to 'abstract'
# data = data.rename(columns={'text': 'abstract'})
#
# data.to_csv('data/newsgroups.csv', index=False)
#
# # make a smaller version with 1000 rows selected randomly
# data_sample = data.sample(1000)
# data_sample['id'] = [str(i) for i in range(1000)]
#
# # save to csv.
# data_sample.to_csv('data/newsgroups_1k_sampled.csv', index=False)

data = pd.read_csv('data/newsgroups_1k_sampled.csv')[['id', 'abstract']]
data['id'] = data['id'].astype(str)

print(data.loc[data['id'] == '10', 'abstract'].iloc[0])
