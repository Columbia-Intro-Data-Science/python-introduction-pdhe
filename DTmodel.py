
import csv
from random import randint
import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans



import graphviz
import matplotlib.pyplot as plt




def open_csvfile(file_name):
    X=open(file_name,"r")
    Xdata=[]
    
    with X:
        reader = csv.reader(X)
        for row in reader:
            #%for e in row:
            Xdata.append(row)
            
    return Xdata


#filename='restaurant_edited.csv'
filename2='data_2.csv'


features=pd.read_csv(filename2,index_col=0,low_memory=False)
#restaurant=pd.read_csv(filename,index_col=0,low_memory=False)


features=features.set_index(features['id'])


stars=features['stars']


features=features.drop(['id'],axis=1)
features=features.drop(['stars'],axis=1)


names=list(features.columns.values)

#merge1=pd.DataFrame({})



#features=features[0:10000]
#stars=stars[0:10000]

#left=features[' \r\r\r        breakfast ']
#right=features[' breakfast ']
#merge1=pd.merge(left.to_frame(),right.to_frame(),how='outer',left_on=True,left_index=True)#,left_on='Index', right_index=True)

#merge1=pd.concat([features,stars], axis=0)
#merge works, but might not needed

features=features.drop(['AgesAllowed'],axis=1)

#features['review_count']=restaurant['review_count']



features=features.fillna(0.5)
features=features.replace(to_replace=' FALSE', value=0)
features=features.replace(to_replace=' TRUE', value=1)
features=features.replace(to_replace=' list', value=0)

features=features.replace(to_replace=' "beer_and_wine"', value=1)
features=features.replace(to_replace=' "none"', value=1)
features=features.replace(to_replace=' "full_bar"', value=2)

features=features.replace(to_replace= ' "casual"', value=1)
features=features.replace(to_replace= ' "dressy"', value=2)
features=features.replace(to_replace= ' "formal"', value=3)

features=features.replace(to_replace= ' "free"', value=1)
features=features.replace(to_replace= ' "no"', value=0)

features=features.replace(to_replace= ' "outdoor"', value=2)
features=features.replace(to_replace= ' "yes"', value=1)
features=features.replace(to_replace= ' "yes_free"', value=12)
features=features.replace(to_replace= ' "yes_corkage"', value=11)

features=features.replace(to_replace= ' TRUE FALSE', value=0)
features=features.replace(to_replace= ' FALSE TRUE', value=0)
features=features.replace(to_replace= ' TRUE TRUE', value=1)
features=features.replace(to_replace= ' FALSE FALSE', value=0)

features=features.replace(to_replace= ' "paid"', value=5)

stars=stars.replace(to_replace= 'continuous', value=0)




features=features.replace(to_replace= ' "average"', value=3)
features=features.replace(to_replace= ' "loud"', value=2)
features=features.replace(to_replace= ' "quiet"', value=4)
features=features.replace(to_replace= ' "very_loud"', value=1)

features=features.replace(to_replace= 'continuous', value=0)




#stars=stars


# try feature selection

#X,y=features,stars
#X_new=SelectKBest(chi2,k=1).fit_transform(X,y)



#Y = features[' breakfast ']
#X = features[[' breakfast ',' HasTV ',' Music ',' NoiseLevel ']]

#
#lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(features.astype('float'), stars.astype('int'))
#model = SelectFromModel(lsvc, prefit=True)
#X_new = model.transform(X)
#X_new.shape
#
#X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_new, stars.astype('int'), test_size=0.2, random_state=0)
#
#

X_train, X_test, y_train, y_test = cross_validation.train_test_split(features.astype('float'), stars.astype('int'), test_size=0.2, random_state=0)

tree = DecisionTreeClassifier(max_depth=8)
RFtree=RandomForestRegressor(n_estimators=50,max_depth=8)
Boosttree=GradientBoostingClassifier(n_estimators=50,max_depth=7)
ADAtree=AdaBoostClassifier(n_estimators=30,learning_rate=1)



score=[]
for dep in range(5,40):
    
    tree = DecisionTreeClassifier(max_depth=dep)
    tree.fit(X_train, y_train)
    score.append(tree.score(X_test, y_test))
    
plt.figure()
plt.title('score of normal decition tree max depth vs score')

plt.plot(range(5,40),score)

# max_depth of 8 is best for normal decision tree

socre=[]
for dep in range(5,40):
    print(dep)
    RFtree=RandomForestRegressor(n_estimators=15,max_depth=dep)
    RFtree.fit(X_train, y_train)
    score=RFtree.score(X_test, y_test)

plt.figure()
plt.title('score of random forest tree max depth vs score')

plt.plot(range(5,40),score)

    
#    



tree = DecisionTreeClassifier(max_depth=8)
tree.fit(X_train, y_train)
tree.score(X_test, y_test)


RFtree.fit(X_train, y_train)
RFtree.score(X_test, y_test)


Boosttree.fit(X_train, y_train)
Boosttree.score(X_test, y_test)


ADAtree.fit(X_train, y_train)
ADAtree.score(X_test, y_test)


#scores = cross_val_score(tree,features.astype('float'),stars.astype('float'), cv=5)

knnscore=[]
for neighbor in range(1,15):
    KNN = KNeighborsClassifier(n_neighbors=neighbor)
    KNN.fit(X_train, y_train) 
    knnscore.append(KNN.score(X_test, y_test))
    
plt.figure()
plt.title('kNN Score vs Neighbors')
plt.plot(range(1,10),knnscore)

#GMM=GaussianMixture(n_components=5)
#GMM.fit(X_train, y_train)
#GMM.score(X_test, y_test)



#Kmeans=KMeans(n_clusters=5,n_init=20, max_iter=500)
#Kmeans.fit(X_train, y_train)
#Kmeans.score(X_test, y_test)
    
from sklearn.neural_network import MLPClassifier
MLP=MLPClassifier()
MLP.fit(X_train, y_train)
MLP.score(X_test, y_test)

from sklearn.linear_model import RidgeClassifier
RRG=RidgeClassifier()
RRG.fit(X_train, y_train)
RRG.score(X_test, y_test)





scores=cross_val_score(tree,features,stars, cv=5,n_jobs=4)

boostscores=cross_val_score(Boosttree,features.astype('float'),stars.astype('int'), cv=5,n_jobs=4)

print ('R^2 = ', scores)



dot_data=export_graphviz(tree, out_file='Reg_Tree.dot')
with open("Reg_Tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph, format='png')


dot_data=export_graphviz(Boosttree, out_file='Reg_Tree.dot')
with open("Reg_Tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph, format='png')

from sklearn.externals import joblib
joblib.dump(tree, 'regulartree.pkl') 
joblib.dump(RFtree, 'RFtree.pkl') 
joblib.dump(Boosttree, 'Boosttree.pkl') 
joblib.dump(ADAtree, 'ADAtree.pkl') 
joblib.dump(KNN, 'KNN.pkl')


#clf = joblib.load('filename.pkl') 

#example
features.to_csv('features.csv')
Xtest=features[0:1]

joblib.dump(Xtest, 'Xtest.pkl') 
Xtest = joblib.load('Xtest.pkl') 

Xtest.to_pickle(Xtest)
df = pd.read_pickle(Xtest)

for col in Xtest.columns:
    
    Xtest.set_value(5,col,1)
    
Ypredict=tree.predict(Xtest)


RestaurantsTakeOut=request.form['select1']
valet=request.form['select2']
street=request.form['select3']
validated=request.form['select4']
RestaurantsAttire=request.form['select5']
#classy=
#trendy=
#touristy=
intimate=request.form['select9']



###cross calidation







