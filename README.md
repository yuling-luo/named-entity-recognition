# named-entity-recognition
Implemented a Bidirectional LSTM-CRF model with attention approach for name entity recognition

## Dataset
Used PART OF the CONLL-2003 from Reuters dataset, which contains a large set of sentences from English news articles with corresponding NER tags for each word. You are provided with three files (train, val, test)

## Data preprocessing
In the section of data preprocessing, I used stemming to reduce words to the root word for sentences in the train, validation and test dataset. In the context of name entity recognition, stemming makes training data denser by reducing the size of dictionary, which could help with performance improvement for the deep learning model. In addition, I added ‘<START>’ and ‘<STOP>’ to NER labels.

## Input Embedding
Used Glove word embedding, POS-Tag embedding and parse tree embedding

## NER model with attention approach
In this project,  I used a Bidirectional LSTM-CRF model with attention approach for the task of name entity recognition. The combination of BI-LSTM and CRF has been shown to be very successful in the field of sequence labeling task in past few years. For embedding layer, we used pre-trained glove word embedding and pos-tag together.

## Evaluation
I evaluated different combination of embedding; different layer strategy and different attention strategy
