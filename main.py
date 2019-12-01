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

    #evaluation_documents, evaluation_labels = NaivBayesClassifier.read_Document("ourtest.txt")




    trainedData,lableProbability = NaivBayesClassifier.train_nb(training_documents,training_label)

    # print(trainedData["pos"])
    # print("\nsize of pos training data", len(trainedData["pos"]))
    # print(trainedData["neg"])
    # print("size of neg training data", len(trainedData["neg"]))
    # print(lableProbability)




    # print(evaluation_labels)
    # print("size of evalutaion_labels", len(evaluation_labels))
    testDocument = "It's just so boring. I fell asleep while playing. It's a walking simulator and a bad one at that. Only reason it didn't get a 0 is because of the graphics. That's it. Kojima is overrated."
    print("testing: ", testDocument)
    print('logarithmic Score for `pos` label = ', NaivBayesClassifier.score_doc_label(testDocument, 'pos' ,trainedData, lableProbability))
    print('logarithmic Score for `neg` label = ', NaivBayesClassifier.score_doc_label(testDocument, 'neg', trainedData, lableProbability))
    print('The document is more probably : ',NaivBayesClassifier.classify_nb(testDocument,trainedData, lableProbability))

    print("\n\n")





    guessedLabels = NaivBayesClassifier.classify_documents(evaluation_documents,trainedData, lableProbability)

    print('Accuracy of classifier :', NaivBayesClassifier.accuracy(evaluation_labels,guessedLabels))


