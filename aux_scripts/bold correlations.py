import pandas as pd
import csv


df = pd.read_csv('/Users/anastasiastefanescu/Documents/dataseturi eyetracking/bold_response_LH.csv')
pred = pd.read_csv('/Users/anastasiastefanescu/Documents/dataseturi eyetracking/pred_timpi_propozitii.csv')
other = pd.read_csv('/Users/anastasiastefanescu/Documents/dataseturi eyetracking/predictii/pred_total_test.csv')
pred_upd = pd.read_csv('/Users/anastasiastefanescu/Documents/dataseturi eyetracking/pred_timpi_propozitii_updated.csv')

columns = df.columns
t = 'total_time_prediction'
#m = 'mean_time_prediction'

with open('corelatii2.csv', 'a', newline='') as csvfile: 
    wrt = csv.writer(csvfile)
    wrt.writerow(['bold_column', 'corr_total_time']) #, 'corr_mean_time'
    for c in columns:
        if c != 'item_id' and c != 'sentence':
            wrt.writerow([c, pred_upd[t].corr(df[c])]) #, pred[m].corr(df[c])
            #print(f"{c} -> total: {pred[t].corr(df[c])} ; mean : {pred[m].corr(df[c])}")

    
    # for c in columns:
    #     if c != 'item_id' and c != 'sentence':
    #         for d in columns:
    #             if d != 'item_id' and d != 'sentence' and d!= c:
    #                 print(f"{c} - {d} -> {df[c].corr(df[d])}")
