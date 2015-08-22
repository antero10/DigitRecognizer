from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from numpy import savetxt


print 'Reading the data...'

#Reading train data set
train = pd.read_csv('train.csv')
#Reading test data set
test = pd.read_csv('test.csv')

#Visualizing train data

print 'Printing train summary'
print train.head()

print 'Printing test summary'
print test.head()

#Random forest

rf = RandomForestClassifier(n_jobs=2)

features = train.columns[1:]

y,_ =  pd.factorize(train['label'])

rf.fit(train[features],y)

preds = [[index + 1,x] for index,x in enumerate(rf.predict(test[features]))]

#pd.crosstab(test['label'], preds, rownames=['actual'], colnames=['preds'])



savetxt('submission.csv', preds, delimiter=',',fmt='%d,%d',
            header='ImageId,Label', comments = '')
print 'File created!'
