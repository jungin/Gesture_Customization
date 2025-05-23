import os
import shutil
import pandas as pd
from variables import data_dir, train_csv, preprocessed_val_dir, val_csv

# 15: Swiping Down
# 9: Shaking Hand
# 12: Sliding Two Finger Right
target = val_csv
df = pd.read_csv(target)
new = [9, 12, 15]

# extract three gestures
new_df = df[df['label_id'].isin(new)]

# files = new_df['video_id'].to_list()
# for file in files:
#     filename = str(file) + '.npy'
#     src = os.path.join(preprocessed_train_dir, filename)
#     dst = os.path.join(data_dir, 'new', 'test', filename)
#     if os.path.isfile(src):
#         os.makedirs(os.path.join(data_dir, 'new', 'test'), exist_ok=True)
#         shutil.copy(src, dst)

# not in the new gestures
not_new = df[~df['label_id'].isin(new)]
files = not_new['video_id'].to_list()
print(len(files))
for file in files:
    filename = str(file) + '.npy'
    src = os.path.join(preprocessed_val_dir, filename)
    dst = os.path.join(data_dir, 'new', 'val', filename)
    if os.path.isfile(src):
        os.makedirs(os.path.join(data_dir, 'new', 'val'), exist_ok=True)
        shutil.copy(src, dst)


