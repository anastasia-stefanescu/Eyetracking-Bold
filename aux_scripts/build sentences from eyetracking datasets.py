import pandas as pd
import csv

inc = '/Users/anastasiastefanescu/Documents/dataseturi eyetracking/'
folder = 'datasets/all_eng/'
train_df = pd.read_csv(inc + folder + 'train_dataset.csv')
test_df = pd.read_csv(inc + folder + 'test_dataset.csv')
val_df = pd.read_csv(inc + folder + 'val_dataset.csv')

dfs = [train_df, test_df, val_df]
files = ['train', 'test', 'val']
for i in range(3):
    df = dfs[i]
    output_file = files[i] + '_sent.csv'
    with open(output_file, 'a', newline='') as csvfile:
        wrt = csv.writer(csvfile)
        wrt.writerow(['id', 'sentence', 'total_fix_dur'])

        lg = len(df)
        sentence = ""
        sum = 0
        for i in range(lg):
            sentence += str(df.loc[i, 'word'])+ " "
            sum += df.loc[i, 'total_fix_dur']
            if i < lg-1 and df.loc[i, 'sentence_num'] != df.loc[i+1, 'sentence_num']:
                wrt.writerow([df.loc[i, 'sentence_num'], sentence, sum])
                sentence = ""
                sum = 0
