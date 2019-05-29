import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from helpers import *
from sentiment_analyzer import SentimentAnalyzer

def main():
    analyzer = initializeAnalyzer()

    df = pd.read_csv("../source/data/data.csv", error_bad_lines=False)

    X_test = df["comments"]
    y_test = df["category"]

    predictions = []
    i = 0
    tonalities = []
    for sentence in X_test:
        sys.stdout.write("\r" + str(i))
        sys.stdout.flush()
        i+=1
        result=analyzer.analyze(sentence)
        tonalities.append(result)
        if result ==0:
            predictions.append("neutral")
        elif result >0:
            predictions.append("positive")
        else:
            predictions.append("negative")

    try:
        print("LOG:\n\tAccuracy score:" + str(accuracy_score(predictions, y_test)))

        print("LOG:\n\tF1 score(average = None):")
        print(f1_score(y_test, predictions, average=None))

        print("LOG:\n\tF1 score(average = macro):")
        print(f1_score(y_test, predictions, average='macro'))

        print("LOG:\n\tF1 score(average = micro):")
        print(f1_score(y_test, predictions, average='micro'))

        print("LOG:\n\tF1 score(average = weighted):")
        print(f1_score(y_test, predictions, average='weighted'))

    except Exception as e:
        print("main.py: Error occured:" + str(e))
    # df["predictions"] = predictions
    # df.to_csv("../source/error606.csv", index = None, header=True)


def test():
    analyzer = initializeAnalyzer()
    while 1==1:
        print("Enter any kazakh written sentence to preceed:")
        result = analyzer.analyze(input())
        print(result)

if __name__ == "__main__":
    main()
    # test()
