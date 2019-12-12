import Model as model
import sklearn.ensemble as ens
import sklearn.model_selection as ms
from sklearn.metrics import classification_report
import seaborn as sb
import matplotlib.pyplot as mp
import pandas as pd


# map function
def map(val):
    if val > 7.5:
        return 0
    elif val > 6:
        return 1
    else:
        return 2


# returns the regression dataset with a mapped rating
def mapped_data():
    data = model.getdata()
    data["mrating"] = data["rating"].apply(lambda x: map(x))
    return data


def random_forest_classifier():
    data = mapped_data()
    data = data.drop(columns=["name"])
    # preparing data
    rfr = ens.RandomForestClassifier()
    # the model
    data_x = data.drop(columns=['rating', 'mrating'])
    data_y = data.drop(columns=['crashcount', 'averagestops', 'delta', 'rating'])
    # define target and predictors
    gs = ms.GridSearchCV(estimator=rfr, param_grid={'n_estimators': [1, 3, 20, 50, 100, 200],
                                                    'criterion': ['entropy', 'gini'],
                                                    'max_depth': [1, 5, 10, 20],
                                                    'min_samples_leaf': [1, 3, 5, 8],
                                                    'max_leaf_nodes': [2, 5, 10, 15, 20]
                                                    }, cv=3, n_jobs=-1)
    # searches best parameters !!!warning takes for ever with this amount of variables and takes up all cpu
    x_train, x_test, y_train, y_test = ms.train_test_split(data_x, data_y, test_size=0.8)
    # test train data
    res = gs.fit(y_train, y_train)
    # search best estimator
    rfr = res.best_estimator_
    rfr.fit(x_train, y_train)
    y_predict = rfr.predict(x_test)
    testpredict = y_test
    testpredict["p_rating"] = y_predict
    # dataframe with true and predict values
    print(classification_report(testpredict["mrating"], testpredict["p_rating"]))
    # classification matrix
    conf = pd.crosstab(testpredict["mrating"], testpredict["p_rating"], rownames=["Actual"], colnames=["Predicted"],
                       margins=True)
    sb.heatmap(conf, annot=True)
    mp.show()
    # confusion matrix


random_forest_classifier()
