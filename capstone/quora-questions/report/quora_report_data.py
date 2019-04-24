import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt



dirpath = os.path.realpath('./input')
filepath_data_train = os.path.join(dirpath, 'train.csv')

tqdm.pandas()
df_train = pd.read_csv(filepath_data_train)
df_train['question_length'] = df_train['question_text'].progress_apply(lambda x: len(x.split()))
#df_outliers = df_train[df_train['question_length'] > 65]
#for index, row in df_outliers.iterrows():
#    print('[' + str(index) + ']', row[0], row[1], row[2], row[3])
quantiles = [1-(0.1**x) for x in range(10)]
quantiles_values = df_train['question_length'].quantile(quantiles)
plt.plot(range(10), quantiles_values)
plt.xticks(range(10), quantiles)
plt.yticks(quantiles_values)
plt.title('Question length percentiles')
plt.show()

#df_train.hist(column='question_length')
#plt.savefig('hist_question_length.png')
#plt.show()

#target_percentage = (100. * len(df_train[df_train['target'] == 1]))/len(df_train)
#print('Insincere percentage:', target_percentage)
#df_train.hist(column='target', bins=[0,1,2])
#plt.margins(1., 0.3)
#plt.savefig('hist_target.png')
#plt.show()
