import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import warnings
from scipy.misc import imread
warnings.filterwarnings('ignore')
from IPython.display import HTML
from collections import Counter

# Importing the dataset
dataset=pd.read_csv('tmdb_5000_movies.csv')
mov=pd.read_csv('tmdb_5000_credits.csv')

# changing the genres column from json to string
dataset['genres']=dataset['genres'].apply(json.loads)
for index,i in zip(dataset.index,dataset['genres']):
    list1=[]
    for j in range(len(i)):
        list1.append((i[j]['name']))# the key 'name' contains the name of the genre
    dataset.loc[index,'genres']=str(list1)

dataset['genres']=dataset['genres'].str.strip('[]')
dataset['genres']=dataset['genres'].str.replace(' ','')
dataset['genres']=dataset['genres'].str.replace("'",'')
dataset['genres']=dataset['genres'].str.split(',')

# changing the production company column from json to string
dataset['production_companies']=dataset['production_companies'].apply(json.loads)
for index,i in zip(dataset.index,dataset['production_companies']):
    list1=[]
    for j in range(len(i)):
        list1.append((i[j]['name']))# the key 'name' contains the name of the genre
    dataset.loc[index,'production_companies']=str(list1)

dataset['production_companies']=dataset['production_companies'].str.strip('[]')
dataset['production_companies']=dataset['production_companies'].str.replace(' ','')
dataset['production_companies']=dataset['production_companies'].str.replace("'",'')
dataset['production_companies']=dataset['production_companies'].str.split(',')

dataset = dataset.join(pd.DataFrame(
    {
        'Action': 0,
        'Action': 0,
        'Adventure': 0,
        'Fantasy': 0,
        'ScienceFiction': 0,
        'Crime': 0,
        'Drama': 0,
        'Thriller': 0,
        'Animation': 0,
        'Family': 0,
        'Western': 0,
        'Comedy': 0,
        'Romance': 0,
        'Horror': 0,
        'Mystery': 0,
        'History': 0,
        'War': 0,
        'Music': 0,
        'Documentary': 0,
        'Foreign': 0,
        'TVMovie': 0
    }, index=dataset.index
))

# filling the genres column
for index,i in zip(dataset.index,dataset['genres']):
    for j in range(len(i)):
        list0 = i[j] 
        dataset.loc[index,list0] = 1
        
# all_pc will contain all the production_companies with repetition
# unique_pc will contain unique production comanies
unique_pc = []
all_pc = []

for index,i in zip(dataset.index,dataset['production_companies']):
    for j in range(len(i)):
        if i[j] not in unique_pc:
            unique_pc.append(i[j])
        all_pc.append(i[j])

# allocating score for the production_companies
dict_pc = {}
count_pc = {}

for j in range(len(unique_pc)):
    dict_pc[unique_pc[j]] = 0;
    count_pc[unique_pc[j]] = 0;
for index,i in zip(dataset.index,dataset['production_companies']):
    for j in range(len(i)):
        dict_pc[i[j]] += dataset['revenue'][index]
        count_pc[i[j]] = count_pc[i[j]] + 1

for j in range(len(unique_pc)):
    dict_pc[unique_pc[j]] = dict_pc[unique_pc[j]]/count_pc[unique_pc[j]]; 

# adding a column pc_score
dataset['pc_score'] = pd.Series(np.random.randn(len(dataset['genres'])), index=dataset.index)

# allocating a production company score for a movie
for index,i in zip(dataset.index,dataset['production_companies']):
    count = 0 
    for j in range(len(i)):
        count += dict_pc[i[j]]
    dataset.loc[index,'pc_score']= count/len(i)

# moving revenue to last
cols = list(dataset.columns.values)
cols.pop(cols.index('revenue')) 
dataset = dataset[cols+['revenue']]

cols = list(dataset.columns.values)
cols.pop(cols.index('original_title')) 
dataset = dataset[['original_title']+cols]

# Faulty Row
dataset.loc[4553, 'release_date'] = '2000-01-01'

# Extracting date of release
release_date_1 = []
for i in range(0,len(dataset)):
    i = dataset.loc[i,'release_date']
    a = 1000*int(i[0]) + 100*(int(i[1])) + 10*(int(i[2])) + int(i[3])
    if(a<1956):
        a = 1956    
    release_date_1.append(a)

# Importing inflation data and making revenue to a base price year 1956
inflation_stuff = pd.read_csv("year_and_inflation.csv")
n1 = len(inflation_stuff)
inflation_stuff.loc[n1-1, 'inflation'] = 1
for i in range(1,n1):
    inflation_stuff.loc[n1 - 1 - i, 'inflation'] =  inflation_stuff.loc[n1-i,'inflation']*(1 + (inflation_stuff.loc[n1 - i - 1, 'inflation']/100))

# adjusting the revenue and budget    
for i in range(0,len(dataset)):
    dataset.loc[i,'revenue'] /= inflation_stuff.loc[2017 - release_date_1[i], 'inflation']
for i in range(0,len(dataset)):
    dataset.loc[i,'budget'] /= inflation_stuff.loc[2017 - release_date_1[i], 'inflation']


# dividing the revenue into 9 classes
r1 = 1000000
r2 = 10000000
r3 = 20000000
r4 = 40000000
r5 = 65000000
r6 = 100000000
r7 = 150000000
r8 = 200000000

# creating a column for revenue class
dataset['revenue_class'] = pd.Series(np.random.randn(len(dataset['revenue'])), index=dataset.index)

# putting the class values got from the range
for index,i in zip(dataset.index,dataset['revenue_class']):
    c = 0;
    val = dataset['revenue'][index];
    if (val <= r1):
        c = 1;
    elif (val > r1 and val <= r2):
        c = 2;
    elif (val > r2 and val <= r3):
        c = 3;
    elif (val > r3 and val <= r4):
        c = 4;
    elif (val > r4 and val <= r5):
        c = 5;
    elif (val > r5 and val <= r6):
        c = 6;
    elif (val > r6 and val <= r7):
        c = 7;
    elif (val > r7 and val <= r8):
        c = 8;
    else:
        c = 9;
    
    dataset.loc[index,'revenue_class']= c
# Extracting Month
month_release = []
for i1 in range(0,len(dataset)):
    i = dataset.loc[i1,'release_date']
    a = 10*(int(i[5])) + int(i[6])
    a = str(a)
    month_release.append(a)

# Adding month of the release
dataset = dataset.join(pd.DataFrame({'Month': '1'},index=dataset.index))
for i in range(0,len(dataset)):
    dataset.loc[i,'Month'] = str(month_release[i])

months = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
for i in months:
    dataset[i] = 0

# filling the month column
for index,i in zip(dataset.index,dataset['Month']):
    for j in range(len(i)):
        list0 = str(i[j]) 
        dataset.loc[index,list0] = 1
        
m = []
import ast
# Converting the dict-like-string to dict
for j in range(0,len(mov['crew'])):
    m.append( ast.literal_eval( mov['crew'][j]) )

mov.crew = m

m = []

# Converting the dict-like-string to dict
for j in range(0,len(mov['cast'])):
    m.append( ast.literal_eval( mov['cast'][j]) )

mov.cast = m
# Copying from here will be the best idea to put this
# code after the revenue has been taken care of.
director_list = []
writer_list = []
editor_list = []
producer_list = []

for i,i1 in zip(mov.crew, mov.index):
    dire = 1
    writ = 1
    edit = 1
    prod = 1
    director_list.append('Someone')
    writer_list.append('Someone')
    editor_list.append('Someone')
    producer_list.append('Someone')
    
    for j in i:
        if (j['job'] == 'Director' and dire):
            if 'name' in list(j.keys()):
              director_list[i1] = j['name']
              dire = 0
        
        elif (j['job'] == 'Editor' and edit):
          if 'name' in list(j.keys()):  
            editor_list[i1] = j['name'] 
            edit = 0
            
        elif (j['job'] == 'Writer' and writ):
          if 'name' in list(j.keys()):
            writer_list[i1] = j['name']
            writ = 0
            
        elif (j['job'] == 'Producer' and prod):
          if 'name' in list(j.keys()):  
            producer_list[i1] = j['name']
            prod = 0
            

def scores(target, revenue, ratings):
    tar_sc = {}
    tar_n = {}
    for i1 in range(0,len(target)):
        i = target[i1]
        j = revenue[i1]
        k = float(ratings[i1])
        if i == 'Someone':
            tar_sc[i] = 0
            continue
        k /= 10;
        if i not in list(tar_sc.keys()):
            tar_sc[i] = j*k
            tar_n[i] = 1
        else:
            tar_sc[i] = (tar_sc[i]*tar_n[i] + j*k)/(tar_n[i] + 1)
            tar_n[i] += 1
    return tar_sc


director_scores1 = scores(director_list, list(dataset.revenue), list(dataset.vote_average))
editor_scores = scores(editor_list, list(dataset.revenue), list(dataset.vote_average))
writer_scores = scores(writer_list, list(dataset.revenue), list(dataset.vote_average))
dataset = dataset.join(pd.DataFrame({
        'Director_score': 0,
        'Editor_score': 0,
        'Writer_score': 0,
        'Cast1': 0,
        'Cast2': 0,
        'Cast3': 0,
        'Cast4': 0,
        'Cast5': 0
        }, index = mov.index))
    
cast1 = []
cast2 = []
cast3 = []
cast4 = []
cast5 = []

for i in range(0,len(mov)): 
    cast1.append('Someone')
    cast2.append('Someone')
    cast3.append('Someone')
    cast4.append('Someone')
    cast5.append('Someone')

for i in range(0,4803):    
    if i == 4553:
        continue
    n = len(mov['cast'][i])    
    if n>=1:
        cast1[i] = mov['cast'][i][0]['name']    
    if n>=2:
        cast2[i] = mov['cast'][i][1]['name']    
    if n>=3:
        cast3[i] = mov['cast'][i][2]['name']    
    if n>=4:
        cast4[i] = mov['cast'][i][3]['name']    
    if n>=5:
        cast5[i] = mov['cast'][i][4]['name']
        
cast1_score = scores(cast1, list(dataset.revenue), list(dataset.popularity/10))                                              
cast2_score = scores(cast2, list(dataset.revenue), list(dataset.popularity/10))
cast3_score = scores(cast3, list(dataset.revenue), list(dataset.popularity/10))
cast4_score = scores(cast4, list(dataset.revenue), list(dataset.popularity/10))
cast5_score = scores(cast5, list(dataset.revenue), list(dataset.popularity/10))

for i in range(0,len(mov)):
    if director_list[i] in list(director_scores1.keys()):
      dataset.loc[i, 'Director_score'] = director_scores1[director_list[i]]
    if editor_list[i] in list(editor_scores.keys()):  
      dataset.loc[i, 'Editor_score'] = editor_scores[editor_list[i]]
    if writer_list[i] in list(writer_scores.keys()):
      dataset.loc[i, 'Writer_score'] = writer_scores[writer_list[i]]
    if cast1[i] in list(cast1_score.keys()):  
      dataset.loc[i, 'Cast1'] = cast1_score[cast1[i]]
    if cast2[i] in list(cast2_score.keys()):
      dataset.loc[i, 'Cast2'] = cast2_score[cast2[i]]
      dataset.loc[i, 'Cast3'] = cast3_score[cast3[i]]
      dataset.loc[i, 'Cast4'] = cast4_score[cast4[i]]
      dataset.loc[i, 'Cast5'] = cast5_score[cast5[i]]       

del dataset['homepage']
del dataset['id']
del dataset['keywords']
del dataset['original_language']
del dataset['overview']
del dataset['production_countries']
del dataset['runtime']
del dataset['spoken_languages']
del dataset['status']
del dataset['tagline']
del dataset['title']
del dataset['genres']
del dataset['production_companies']
del dataset['release_date']
del dataset['revenue']
del dataset['Month']

# moving revenue to last
cols = list(dataset.columns.values)
cols.pop(cols.index('revenue_class')) 
dataset = dataset[cols+['revenue_class']]

dataset.drop(dataset.columns[-10], axis=1, inplace = True)
dataset.drop(dataset.columns[-24], axis=1, inplace = True)
 
# dependent and inpendent variables
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:,:])
X[:,:] = imputer.transform(X[:,:])

from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
X = sel.fit_transform(X)

l = len(X[0])
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
y_train = y_train.reshape(-1, 1)
labelencoder_y_train = LabelEncoder()
y_train[:, 0] = labelencoder_y_train.fit_transform(y_train[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
y_train = onehotencoder.fit_transform(y_train).toarray()

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
y_test = y_test.reshape(-1, 1)
labelencoder_y_test = LabelEncoder()
y_test[:, 0] = labelencoder_y_test.fit_transform(y_test[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
y_test = onehotencoder.fit_transform(y_test).toarray()


# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 15, init = 'uniform', activation = 'relu', input_dim = l))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 15, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 9, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Convert labels to categorical one-hot encoding
# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 500)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred_max_index = np.argmax(y_pred, axis = 1)
y_test_max_index = np.argmax(y_test, axis = 1)

# testing training set accuracy
y_pred_2 = classifier.predict(X_train)
y_pred_2_max_index = np.argmax(y_pred_2, axis = 1)
y_train_max_index = np.argmax(y_train, axis = 1)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_max_index, y_pred_max_index)

from sklearn.metrics import accuracy_score
test_ac = accuracy_score(y_test_max_index, y_pred_max_index) 

from sklearn.metrics import accuracy_score
train_ac = accuracy_score(y_train_max_index, y_pred_2_max_index)

