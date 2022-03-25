import pandas as pd


train_csv = 'processedData/trainingDataF.csv'
df = pd.read_csv(train_csv)
corr_df = df.corr(method='pearson')

print(corr_df['trial_result'])
