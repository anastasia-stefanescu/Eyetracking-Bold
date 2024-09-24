import pandas as pd
import csv

inc = '/Users/anastasiastefanescu/Documents/dataseturi eyetracking/'
folder = 'datasets/zuco/'
output_file_test = "pred_zuco_test.csv"

####afisam corelatia cu datele de test
test_pred = pd.read_csv(inc + output_file_test)
test = pd.read_csv(inc + folder + 'test_dataset.csv')

print(test_pred['prediction'].corr(test['total_fix_dur']))

#####calculam suma pe propozitii a timpilor prezisi pt bold

df = pd.read_csv(inc + 'bold_response_LH.csv')
pred = pd.read_csv(inc + 'pred_zuco_bold.csv')

file_for_sentences = 'pred_timpi_prop_zuco.csv'
with open(file_for_sentences, 'a', newline='') as csvfile: 
    sentences = df['sentence'].tolist()
    n = len(sentences)
    indexed = []
    wrt = csv.writer(csvfile)
    wrt.writerow(['sent_id', 'sentence', 'total_time_prediction'])
    for i in range(n):
        words = sentences[i].split(" ")
        m = len(words)
        sum_total = 0
        for j in range(m):
            sum_total += pred.loc[i, 'prediction']
        wrt.writerow([i, sentences[i], sum_total])

df = pd.read_csv(inc + 'bold_response_LH.csv')
pred_upd = pd.read_csv(inc + file_for_sentences)

columns = df.columns
t = 'total_time_prediction'

with open('corelatii_cu_zuco.csv', 'a', newline='') as csvfile: 
    wrt = csv.writer(csvfile)
    wrt.writerow(['bold_column', 'corr_total_time']) #, 'corr_mean_time'
    for c in columns:
        if c != 'item_id' and c != 'sentence':
            wrt.writerow([c, pred_upd[t].corr(df[c])]) #, pred[m].corr(df[c])
            print(f"{c} -> total: {pred_upd[t].corr(df[c])}")
