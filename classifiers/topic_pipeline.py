from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

import pickle
from os.path import exists
from pathlib import Path



home = str(Path.home())
project_folder = r'\parroty_workspace'
output_folder = home + project_folder
pipeline_filename = output_folder + 'parroty_pipeline.pickle'

#Load dataset
categories = ['alt.atheism','soc.religion.christian','comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
#print(twenty_train.target_names)


#Tokenizing Text
count_vect = CountVectorizer()
##X_train_counts = count_vect.fit_transform(twenty_train.data)
#print(count_vect.vocabulary_.get(u'algorithm'))

#TfidfTransformer
tfidf_transformer = TfidfTransformer()
##X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#print(X_train_Tfidf.shape)

#Create classifier
##clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

docs_new = ['God is love', 'OpenGL on the GPU is fast']
##X_new_counts = count_vect.transform(docs_new)
##X_new_tfidf = tfidf_transformer.transform(X_new_counts)

##predicted = clf.predict(X_new_tfidf)

#for doc, category in zip(docs_new, predicted):
#    print('%r => %s' % (doc, twenty_train.target_names[category]))


text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB())
])

text_clf.fit(twenty_train.data, twenty_train.target)

predicted_pipeline = text_clf.predict(docs_new)


##TODO: if file exists, load file instead of train
output_file = pickle.dumps(predicted_pipeline)




for doc, category in zip(docs_new, predicted_pipeline):
    print('%r => %s' % (doc, twenty_train.target_names[category]))
