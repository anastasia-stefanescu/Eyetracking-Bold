import pandas as pd
import csv

df = pd.read_csv('/Users/anastasiastefanescu/Documents/dataseturi eyetracking/bold_response_LH.csv')
preds_total = pd.read_csv('/Users/anastasiastefanescu/Documents/dataseturi eyetracking/pred_total_3.csv')
preds_mean = pd.read_csv('/Users/anastasiastefanescu/Documents/dataseturi eyetracking/pred_mean_3.csv')
pred_total_update = pd.read_csv('/Users/anastasiastefanescu/Documents/dataseturi eyetracking/pred_total_bold_updated.csv')

with open('pred_timpi_propozitii_updated.csv', 'a', newline='') as csvfile: 
    sentences = df['sentence'].tolist()
    n = len(sentences)
    indexed = []
    wrt = csv.writer(csvfile)
    wrt.writerow(['sent_id', 'sentence', 'total_time_prediction', 'mean_time_prediction'])
    for i in range(n):
        words = sentences[i].split(" ")
        m = len(words)
        sum_total = 0
        sum_mean = 0
        for j in range(m):
            sum_total += pred_total_update.loc[i, 'prediction']
            #sum_mean += pred_total_update.loc[i, 'prediction']
        wrt.writerow([i, sentences[i], sum_total, sum_mean])

    
    
