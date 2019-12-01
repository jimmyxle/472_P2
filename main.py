import NaivBayesClassifier
import sys
if __name__ == '__main__':

    print(sys.argv)

    all_documents, all_labels = NaivBayesClassifier.read_Document("dataFile.txt")
    split_point = int(0.80 * len(all_documents))
    training_documents = all_documents[:split_point]
    training_label = all_labels[:split_point]
    training_label_unique = list(set(training_label))
    evaluation_documents = all_documents[split_point:]
    evaluation_labels = all_labels[split_point:]


    trainedData, label_probability = NaivBayesClassifier.train_nb(training_documents, training_label)

    if sys.argv > 1:
        print("testing file ", sys.argv[1])
        test_doc, test_label = NaivBayesClassifier.read_Document(sys.argv[1])
        print('logarithmic Score for `pos` label = ',
              NaivBayesClassifier.score_doc_label(test_doc, 'pos', trainedData, label_probability))
        print('logarithmic Score for `neg` label = ',
              NaivBayesClassifier.score_doc_label(test_doc, 'neg', trainedData, label_probability))

        guessedLabels = NaivBayesClassifier.classify_documents(test_doc, trainedData, label_probability)

        print('Accuracy of classifier :', NaivBayesClassifier.accuracy(test_label, guessedLabels))







