import pandas as pd
#pandas is used for data manipulation and data analysis
from sklearn.model_selection import train_test_split
#split the data for training and testing
from sklearn.linear_model import LogisticRegression
#LogisticR and LinearR comes under supervised
import pickle

data=pd.DataFrame({
    'cgpa':[6,7,8,9,5,7.5,8.5,6.5],
    'aptitude':[60,70,80,90,50,75,85,65],
    'communication':[5,6,7,8,4,6,7,5],
    'projects':[1,2,3,4,1,2,3,2],
    'placed':[0,1,1,1,0,1,1,0]
})

X=data[['cgpa','aptitude','communication','projects']]
y=data['placed']

model=LogisticRegression()
model.fit(X,y)

#save model
pickle.dump(model,open('model.pkl','wb'))
print("model trained and saved")

