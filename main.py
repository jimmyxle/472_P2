import NaivBayesClassifier
import sys
if __name__ == '__main__':

    all_documents, all_labels = NaivBayesClassifier.read_Document("dataFile.txt")
    split_point = int(0.80 * len(all_documents))
    training_documents = all_documents[:split_point]
    training_label = all_labels[:split_point]
    training_label_unique = list(set(training_label))
    evaluation_documents = all_documents[split_point:]
    evaluation_labels = all_labels[split_point:]

    trainedData, lableProbability = NaivBayesClassifier.train_nb( training_documents, training_label )

    testDocument="rien a dire si ce n'est que too $hort et ses potes รงa claque c'est tout"
    #print(trainedData)
    #print(str(trainedData), file=open("pOfWordInLabel.txt", "a" ))
    #print(lableProbability)
    print("Testing our classifier with a small sample review : ", testDocument)
    print('logarithmic Score for `pos` label = ', NaivBayesClassifier.score_doc_label(testDocument, 'pos' ,trainedData, lableProbability))
    print('logarithmic Score for `neg` label = ', NaivBayesClassifier.score_doc_label(testDocument, 'neg', trainedData, lableProbability))

    print("\n")
    print('This testing document ',testDocument,' \n is more probably : ',NaivBayesClassifier.classify_nb(testDocument,trainedData, lableProbability))
    print( "\n" )

    # Testing classify_documents(evaluatingDocs ,trainedData, lableProbabilty) function that Classifies a new document
    guessedLabels = NaivBayesClassifier.classify_documents(evaluation_documents,trainedData, lableProbability)


    print('Accuracy of classifier :', NaivBayesClassifier.accuracy(evaluation_labels,guessedLabels))

    print('Accuracy of detecting a positive class:', NaivBayesClassifier.accuracyOfClassDetection(evaluation_labels, guessedLabels , 'pos'))
    print( 'Accuracy of detecting a negative class:', NaivBayesClassifier.accuracyOfClassDetection( evaluation_labels, guessedLabels, 'pos' ) )

    print('Accuracy of guessed positive classes:', NaivBayesClassifier.accuracyOfGuessedClass( evaluation_labels, guessedLabels, 'neg' ) )
    print( 'Accuracy of guessed negative classes:', NaivBayesClassifier.accuracyOfGuessedClass( evaluation_labels, guessedLabels, 'neg' ) )



