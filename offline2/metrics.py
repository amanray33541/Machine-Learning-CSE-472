"""
Refer to: https://en.wikipedia.org/wiki/Precision_and_recall#Definition_(classification_context)
"""


def accuracy(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
    return correct / len(y_true)
    # return None
    

def precision_score(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement
    true_positives = 0
    false_positives = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i] == 1:
            true_positives += 1
        elif y_pred[i] == 1:
            false_positives += 1

    precision = true_positives / (true_positives + false_positives)

    return precision


def recall_score(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement
    tp = 0
    fn = 0

    for i in range(len(y_true)):
        if y_true[i] == y_pred[i] and y_true[i] == 1:
            tp += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            fn += 1

    return tp / (tp + fn)


def f1_score(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    f1 = 2 * (precision * recall) / (precision + recall)

    return f1
