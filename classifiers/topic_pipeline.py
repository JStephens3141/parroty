from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

import pickle
from pathlib import Path


environment = 'test'
home = str(Path.home())
project_folder = r'\parroty_workspace'
output_folder = home + project_folder
pipeline_filename = output_folder + r'\parroty_pipeline.pickle'

#Load dataset
categories = ['alt.atheism','soc.religion.christian','comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
#print(twenty_train.target_names)


#pipeline cache
if Path(pipeline_filename).is_file():
    print('Cached!')
    with open(pipeline_filename, 'rb') as input:
        text_clf = pickle.load(input)
else:

    #Tokenizing Text
    count_vect = CountVectorizer()

    tfidf_transformer = TfidfTransformer()

    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB())
    ])

    text_clf.fit(twenty_train.data, twenty_train.target)

    with open(pipeline_filename, 'wb') as output:
        pickle.dump(text_clf, output)

docs_new = ['God is love', 'OpenGL on the GPU is fast']
predicted_pipeline = text_clf.predict(docs_new)

for doc, category in zip(docs_new, predicted_pipeline):
    print('%r => %s' % (doc, twenty_train.target_names[category]))
