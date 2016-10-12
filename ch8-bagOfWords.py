import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# divide text into 1-grams (use ngram_range=2,2 for 2-grams)
count = CountVectorizer(ngram_range=(1,1))

docs = np.array([
        'The sun is shining',
        'The weather is sweet',
        'The sun is shining and the weather is sweet'])

bag = count.fit_transform(docs)
print(count.vocabulary_)
print(bag.toarray())

import re
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) +  ''.join(emoticons).replace('-', '')
    return text



