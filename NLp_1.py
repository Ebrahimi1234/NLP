# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 22:17:31 2024

@author: Reza
"""

from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns

corpus = ['Time flies flies like an arrow.',
          'Fruit files like a banana.']

one_hot_vectorizer = CountVectorizer(binary=True)
one_hot = one_hot_vectorizer.fit_transform(corpus).toarray()
sns.heatmap(one_hot,annot=True,
            cbar=False, xticklabels=['Sentence 1'],
            yticklabels=['Sentence 2'])
