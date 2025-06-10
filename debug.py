import pandas as pd

# Parquet file locations for training
#/ssh:jwiseman@wukong.cs.nor.ou.edu:/home/student/jwiseman/scr/RAGEN/data/frozenlake:

# Read the Parquet file
df = pd.read_parquet("data/frozenlake/train.parquet")
#df = pd.read_parquet("data/frozenlake/test.parquet")

# Print all columns (disable truncation)
pd.set_option('display.max_columns', None)

# Show the first 10 rows
#print(df.head(10))
print('-----------------------------')
# Print the full row content with nested dicts/lists fully expanded
import pprint
for i in range(3):
    pprint.pprint(df.iloc[500+i].to_dict())
