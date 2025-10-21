import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
data = pd.read_csv('spam.csv',encoding = 'latin-1')
data = data[['v1','v2']] #dropping useless columns and keeping useful columns
data.columns = ['category','message']
print(data.head())
print(data.shape) #for number of rowsa nd columns
data.drop_duplicates(inplace =  True) #drop duplicates
print(data.shape)
#checking null values
print(data.isnull().sum()) 
#replacing ham into not spam
data['category'] = data['category'].replace(['ham','spam'],['not spam','spam'])
mess = data['message']
cat = data['category']
print(data.head())
(mess_train,mess_test,cat_train,cat_test) = train_test_split(mess,cat,test_size=0.2)
cv = CountVectorizer(stop_words = 'english')
cv.fit_transform(mess_train)
features = cv.fit_transform(mess_train)
#creating our model
model = MultinomialNB()
model.fit(features,cat_train)
#test our model
features_test = cv.transform(mess_test)
print(model.score(features_test,cat_test))
#predicting in real time
message = cv.transform(['congratulations,you won lottery']).toarray()
result = model.predict(message)
print(result)