

import pandas as pd
import numpy as np



filename='restaurant_edited.csv'


rawdata=pd.read_csv(filename,index_col=0)

attribute=rawdata['categories']

i=5


att=attribute[5]
#att=attribute

def SPLIT(att):
    
    att=att.split(', ')
    #att=[i.split('(', 1)[0] for i in att]
    #att2=[i.split(')', 1)[0] for i in att]
    #att2=[i.split('=', 1)[0] for i in att2]

    return att

def SPLIT2(att):
    
    att=att.split(',')
    att=[i.split('(', 1)[0] for i in att]
    att2=[i.split(')', 1)[0] for i in att]

    return att2

#find every attribute
totalfeatures=[]
k=0

for sample in attribute:
    features=sample[2:-1]
    features=SPLIT(features)
    for singlef in features:
        if singlef in totalfeatures:
            k+=1
        else:
            totalfeatures.append(singlef)
    

array=np.asarray(totalfeatures)

data=pd.DataFrame({})
for feat in totalfeatures:
    data[feat]=1


att=attribute[6]

#ind=attribute.index'

i=0
for att in attribute:
    prop=SPLIT(att[2:-1])
    dummy={}
    #prop.remove('list')
    #oneind=attribute[ind[i]].index
    for element in prop:
        #BK=[element.split('=')]
        dummy[element]=int(1)
    
    dummydf=pd.DataFrame(dummy,index=[attribute.index[i]])
    
    #data = pd.join([dummydf,data], axis=1,join='ourter')
    data=data.append(dummydf)
    i+=1
#    if i == 30:
#        break
    print(i)
    
names=list(data.columns.values)

data.to_csv('style.csv')



    #dummydf=pd.DataFrame.from_dict(dummy,orient='index')
    #dummydf=pd.DataFrame.T
    
        

        
    
        
        
    


                
    
    
    
    

