import pandas as pd

df = pd.read_csv('adult_race_comb.txt',sep='\t')


df.to_csv('b.csv')