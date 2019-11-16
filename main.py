import NaivBayesClassifier

if __name__ == '__main__':

    all_documents, all_labels = NaivBayesClassifier.read_Document("ourtest.txt")
    split_point = int(0.80 * len(all_documents))
    training_documents = all_documents[:split_point]
    training_label = all_labels[:split_point]
    training_label_unique = list(set(training_label))
    evaluation_documents = all_documents[split_point:]
    evaluation_labels = all_labels[split_point:]

    #test train_nb(documents,labels)
    trainedData,lableProbability = NaivBayesClassifier.train_nb(training_documents,training_label)
    print(NaivBayesClassifier.score_doc_label('book your' , 'SPAM' ,trainedData, lableProbability))







