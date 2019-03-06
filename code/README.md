# Code
The algorithm that gave me the best result is the SGD.

I used the Tune_Model.ipynb to tune my model with GridsearchCV.

# Explanation

### Word tokenize

Tokenize words,allows to separate the words.

Exemple :
```from nltk.tokenize import word_tokenize

example = ['Mary had a little lamb']
     
tokenized_sents = [word_tokenize(i) for i in example]
            
for i in tokenized_sents:
  print i
 
['Mary', 'had', 'a', 'little', 'lamb']
```

### TF-IDF

This statistical measure makes it possible to evaluate the importance of a term contained in a document, relative to a collection or a corpus. The weight increases proportionally to the number of occurrences of the word in the document. It also varies according to the frequency of the word in the corpus.

```
TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document)
IDF(t) = log_e(Total number of documents / Number of documents with term t in it).
Value = TF * IDF
```

### Why clean text data ?
In the text analysis, we find punctuation (!,;:.?), lowercase and uppercase letters and also different signs (@ # ""). 
The purpose of removing these different elements is to make sure that it does not bias for results
