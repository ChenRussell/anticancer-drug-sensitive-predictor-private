import pandas as pd
import numpy as np

df = pd.DataFrame(np.arange(6).reshape(2,3), index=['AA', 'BB'], columns=['one', 'two', 'three'])
print(df)

df2 = df.stack()
print(df2)

df3 = df2.unstack(0)
print(df3)