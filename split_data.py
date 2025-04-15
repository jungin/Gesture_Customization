import pandas as pd
from variables import train_csv

# 15: Swiping Down
# 9: Shaking Hand
# 12: Sliding Two Finger Right

df = pd.read_csv(train_csv)
new = [9, 12, 15]

# extract three gestures
new_df = df[df['label_id'].isin(new)]




