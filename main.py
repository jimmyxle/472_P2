import NaivBayesClassifier

if __name__ == '__main__':

    all_documents, all_labels = NaivBayesClassifier.read_Document("dataFile.txt")
    split_point = int(0.80 * len(all_documents))
    training_documents = all_documents[:split_point]
    training_label = all_labels[:split_point]
    evaluation_documents = all_documents[split_point:]
    evaluation_labels = all_labels[split_point:]

    #test train_nb(documents,labels)
    probability_labels,probability_neg,probability_pos = NaivBayesClassifier.train_nb(training_documents,training_label )
    print("probability of poitive labele: ",probability_labels['pos'])
    print("probability of negative labele: ",probability_labels['neg'] )
    print("probability of positive reviews for word album: p(token=album|review=positive)= ",probability_pos['album'])
    print("probability of negative reviews for word album : ",probability_neg['album'])








