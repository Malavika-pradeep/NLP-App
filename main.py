import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer


sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")

simple_train= ['call you tonight', 'Call me a cab', 'Please call me...PLEASE!']

vect= CountVectorizer()
vect.fit(simple_train)
print(vect.get_feature_names_out())

#transform training data into a document-term matrix
simple_train_dtm= vect.transform(simple_train)
print(simple_train_dtm)
#convert sparse matrix to a dense matrix
print(simple_train_dtm.toarray())


df = pd.DataFrame(simple_train_dtm.toarray(), columns= vect.get_feature_names_out())
print('TRAIN DATAFRAME:')
print(df)

simple_test= ["please don't call me"]
simple_test_dm= vect.transform(simple_test)
simple_test_dm.toarray()

test_df= pd.DataFrame(simple_test_dm.toarray(), columns= vect.get_feature_names_out())

print('TEST DATAFRAME:')
print(test_df)
