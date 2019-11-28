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



    #testDocument = 'error nightmare'
    #testDocument = 'this flash is the perfect back-up for a studio , or main / flash fill for an amateur studio . i use several w / slaves and lines and the images rival set-ups that cost thousands of dollars . i like that they are light and easy to pack and transport . they are reasonably priced . they have been around for a long time and do not have bells and whistles . the sunpak has auto / f-stop and manual . if you are a real photographer , you will know how to use them . the down side is that if your sunpak is plugged into a wall , it takes far too long to recharge ( sometimes 3-5 seconds ) . rechargeable batteries seem to work the best with 1-3 second recharge times between flashes .'
    testDocument="rien a dire si ce n'est que too $hort et ses potes รงa claque c'est tout"
    trainedData,lableProbability = NaivBayesClassifier.train_nb(training_documents,training_label)
    #print(trainedData)
    #print(str(trainedData), file=open("pOfWordInLabel.txt", "a" ))

    #print(lableProbability)

    print('logarithmic Score for `pos` label = ', NaivBayesClassifier.score_doc_label(testDocument, 'pos' ,trainedData, lableProbability))
    print('logarithmic Score for `neg` label = ', NaivBayesClassifier.score_doc_label(testDocument, 'neg', trainedData, lableProbability))
    #
    #
    # print('The document is more probably : ',NaivBayesClassifier.classify_nb(testDocument,trainedData, lableProbability))

    guessedLabels = NaivBayesClassifier.classify_documents(evaluation_documents,trainedData, lableProbability)


    print('Accuracy of classifier :', NaivBayesClassifier.accuracy(evaluation_labels,guessedLabels))


