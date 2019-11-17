import NaivBayesClassifier

if __name__ == '__main__':

    all_documents, all_labels = NaivBayesClassifier.read_Document("dataFile.txt")
    split_point = int(0.80 * len(all_documents))
    training_documents = all_documents[:split_point]
    training_label = all_labels[:split_point]
    training_label_unique = list(set(training_label))
    evaluation_documents = all_documents[split_point:]
    evaluation_labels = all_labels[split_point:]

    #test train_nb(documents,labels)

    # testDocument = 'error nightmare'
    trainedData,lableProbability = NaivBayesClassifier.train_nb(training_documents,training_label)
    # print('log score for `pos` label = ', NaivBayesClassifier.score_doc_label(testDocument , 'pos' ,trainedData, lableProbability))
    # print('log score for `neg` label = ', NaivBayesClassifier.score_doc_label( testDocument, 'neg', trainedData, lableProbability))
    #
    #
    # print('The document is more probably : ',NaivBayesClassifier.classify_nb(testDocument,trainedData, lableProbability))

    guessedLabels = NaivBayesClassifier.classify_documents(evaluation_documents,trainedData, lableProbability)

    print('Accuracy of classifier :', NaivBayesClassifier.accuracy(evaluation_labels,guessedLabels))


