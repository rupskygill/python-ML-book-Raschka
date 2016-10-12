import pandas as pd
from io import StringIO

csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

# If you are using Python 2.7, you need
# to convert the string to unicode:
# csv_data = unicode(csv_data)

df = pd.read_csv(StringIO(csv_data))

# Show count of missing data
df.isnull().sum()

# Drop rows with missing data
df.dropna()

# Drop column if NaN in colum
df.dropna(axis=1)

# only drop rows where all columns are NaN
df.dropna(how='all')

# drop rows that have not at least 4 non-NaN values
df.dropna(thresh=4)

# only drop rows where NaN appear in specific columns (here: 'C')
df.dropna(subset=['C'])

