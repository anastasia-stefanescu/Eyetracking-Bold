import pandas as pd

test_pred_init = pd.read_csv('/Users/anastasiastefanescu/Documents/dataseturi eyetracking/predictii/pred_total_test.csv')
test_pred_upd = pd.read_csv('/Users/anastasiastefanescu/Documents/dataseturi eyetracking/pred_total_test_updated.csv')
test = pd.read_csv('/Users/anastasiastefanescu/Documents/dataseturi eyetracking/test_dataset.csv')

train_rez = pd.read_csv('/Users/anastasiastefanescu/Documents/dataseturi eyetracking/predictii/pred_total.csv')
train_act = pd.read_csv('/Users/anastasiastefanescu/Documents/dataseturi eyetracking/train_dataset.csv')

#0.71 - corelatia dintre predictiile pe timpul total si timpul total actual pe datele de test
##print(test['prediction'].corr(test['actual'])) 

# 0.69 - corelatia dintre predictiile pe timpul total si timpul total actual pe datele de train
#print(train_rez['prediction'].corr(train_act['total_fix_dur']))

#0.5 corelatia dintre predictiile de timp total pe datele de test (prezise si cele adevarate)
print(test_pred_upd['prediction'].corr(test['total_fix_dur']))