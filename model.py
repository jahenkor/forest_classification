import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
from sklearn import metrics
from sklearn.tree import export_graphviz
from sklearn.preprocessing import Normalizer
from sklearn.externals.six import StringIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from IPython.display import Image
from sklearn.model_selection import RepeatedKFold
import pydotplus
import matplotlib.pyplot as plt
import seaborn as sns


applyModel = False

def main():
    classifiers = ['decisiontree', 'neuralnetwork', 'bayesian', 'randomforest']
    classifier = 'bayesian'
    predictions = []
    class_labels = [1, 2, 3, 4, 5, 6, 7]
    feature_namess = []

    for x in (range(54)):
        feature_namess.append(x)



    train, test, actualTestLabels, actualTrainLabels = loadTrainData()
    
    #Use test data from file
    if(applyModel):
        test = loadTestData()
    test = np.array(test.values)
    # Test method, change this to loop and test all classifiers using K-fold cross validation
    print(train)
    print(test)
    if (classifier == 'decisiontree'):
        DecisionTreeClf(train, actualTrainLabels, test, class_labels, feature_namess,
                       actualTestLabels)
    if (classifier == 'bayesian'):
        NaiveBayesClassifier(train, actualTrainLabels, test, class_labels, actualTestLabels) 
    if (classifier == 'neuralnetwork'):
        NeuralNetworkClassifier(train, actualTrainLabels, test, class_labels, actualTestLabels)
    if(classifier == 'randomforest'):
        RandomForestClf(train, actualTrainLabels, test, actualTestLabels, class_labels, feature_namess)
    # Print test predictions
    outTestSet(predictions)

    return 0

def RandomForestClf(train, actualTrainLabels, test, actualTestLabels, class_labels, feature_namess):

    predictions = []
    f1_trees = []
    for estimator in range(100,1000):
        i = 0
        rf_clf = RandomForestClassifier(n_estimators=200)
        rf_clf.fit(train, actualTrainLabels)
        for test_inst in test:
            i += 1
            print("Iteration: %d" % i)
            predictions.append(rf_clf.predict(test_inst.reshape(1,-1))[0])
        if(not(applyModel)):
            f1_trees.append(f1_score(actualTestLabels, predictions, class_labels, average='weighted'))
        #print("f1 score: %f" % f1)
        print("feature importance")

        feature_imp = pd.Series(rf_clf.feature_importances_, index=feature_namess).sort_values(ascending=False)
        predictions.clear()
        #visualizeFeatures(feature_imp, feature_imp.index)   


    print("Number of trees test")
    print(depthTest)
    plt.plot(num_depth, depthTest)
    plt.xlabel('Number of trees')
    plt.ylabel('f1 score')
    plt.show()





    return 0

def NaiveBayesClassifier(train, actualTrainLabels, test, class_labels, actualTestLabels):
    # Assume distribution to be normal
    predictions = []

    nb_clf = GaussianNB()
    print(train)
    #print(norm_train)
    nb_clf.fit(train, actualTrainLabels)
    for test_inst in test:
        predictions.append(nb_clf.predict(test_inst.reshape(1,-1))[0])
    if(not(applyModel)):
        f1 = f1_score(actualTestLabels, predictions, class_labels, average='weighted')
    print("Variance of each feature\n")
    print(nb_clf.sigma_)
    print("f1 score: %f" % f1)



    return 0


def DecisionTreeClf(train, actualTrainLabels, test, class_labels, feature_namess, actualTestLabels):
    #print("Number of features in tree: %d " % dectree_clf.n_features_)
    #print("instance is : ", inst_predict)
    i = 0
    predictions = []
    num_depth = []
    for x in range(5,1000):
        num_depth.append(x)

    depthTest = []
    max_depths = 1000
    for depth in range(5, max_depths):
        print("Iter: %d" % depth)
        dectree_clf = DecisionTreeClassifier(max_features=30, criterion="entropy", max_depth = depth)

        dectree_clf.fit(train, actualTrainLabels)
        for test_inst in test:
            print("Iter : %d" % ++i)
    #print(test_inst)
            predictions.append(dectree_clf.predict(test_inst.reshape(1, -1))[0])
            #        ViewTree(dectree_clf, feature_namess, class_labels)
        if(not(applyModel)):
            f1 = f1_score(actualTestLabels, predictions, class_labels, average='weighted')
        predictions.clear()
        depthTest.append(f1)
        print("f1 score: %f" % f1)


    print("depth test")
    print(depthTest)
    plt.plot(num_depth, depthTest)
    plt.xlabel('Depth')
    plt.ylabel('f1 score')
    plt.show()

    feature_selection = pd.Series(dectree_clf.feature_importances_, index=feature_namess).sort_values(ascending=True)
    visualizeFeatures(feature_selection,feature_selection.index)

    # Tree Visualizer
    #      dot_data = tree.export_graphviz(dectree_clf, out_file=None,
    #                                     feature_names = feature_names,
    #                                    class_names = class_labels,
    #                                   filled=True, rounded=True,
    #                                  special_characters=True)
    # graph = graphviz_Source(dot_data)
    # graph

    

    return 0


def NeuralNetworkClassifier(train, actualTrainLabels, test, class_labels, actualTestLabels):

    predictions = []
    neural_clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(20,), random_state=1,
                               activation='logistic',learning_rate='adaptive', shuffle=True, max_iter=1000)
    neural_clf.fit(train, actualTrainLabels)

    for test_inst in test:
        predictions.append(neural_clf.predict(test_inst.reshape(1,-1))[0])
    if(not(applyModel)):
        f1 = f1_score(actualTestLabels, predictions, class_labels, average='weighted')
    predictions.clear()
    print("f1 score: %f" % f1)
        

    return 0


def loadTrainData():
    forestData = pd.read_csv('1551366200_25631_train.csv', sep=',', header=None)
    print(forestData)
    train, test = train_test_split(forestData, train_size=0.95, test_size=0.05, shuffle=True)

    test_classLabels = np.array(test[test.columns[54]])
    train_classLabels = np.array(train[train.columns[54]])

    # delete class labels from train and test

    train.drop(train.columns[54], axis=1, inplace=True)  # column_name: 54, 0 = rows, 1 columns
    test.drop(test.columns[54], axis=1, inplace=True)
#    train.drop(train.columns[36], axis=1, inplace=True)  # column_name: 54, 0 = rows, 1 columns
#    test.drop(test.columns[36], axis=1, inplace=True)
#    train.drop(train.columns[43], axis=1, inplace=True)  # column_name: 54, 0 = rows, 1 columns
#    test.drop(test.columns[43], axis=1, inplace=True)
#    train.drop(train.columns[42], axis=1, inplace=True)  # column_name: 54, 0 = rows, 1 columns
#    test.drop(test.columns[42], axis=1, inplace=True)





    print("Class Labels Test: \n")
    print(test_classLabels)
    print("Class Labels Train: \n")
    print(train_classLabels)

    #    rkf = RepeatedKFold(n_splits=5, n_repeats=5)
    #    for train,test in rkf.split(forestData):
    #        print("train: %s test: %s" % (train[0][54], test[0][54]))

    return train, test, test_classLabels, train_classLabels


def loadTestData():
    test_dataset = pd.read_csv("1551366200_2625306_test_no_label.csv", sep=",", engine="python", header=None)
    test_dataset.drop(test_dataset.columns[36], axis=1, inplace=True)
    test_dataset.drop(test_dataset.columns[43], axis=1, inplace=True)


    return test_dataset


def Prediction(actual, prediction):
    # Confusion matrix
    return 0


def outTestSet(predictions):
    with open('dectree.dat', 'w') as file:
        for i in range(len(predictions)):
            file.write("%s\n" % predictions[i])
        file.flush()

# Credit author
def visualizeFeatures(features, index):

    #%matplotlib inline

    sns.barplot(y=features, x=index)

    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.title('Features according to importance')
    plt.legend()
    plt.show()

# Change this, stealing source!
def ViewTree(clf, features, classes):
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names=features,
                    class_names=['1', '2', '3', '4', '5', '6', '7'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('forest.png')
    Image(graph.create_png())


main()
