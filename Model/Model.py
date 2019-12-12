import DataAccess.DataAccess as da
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.tree import export_graphviz
from IPython.display import Image
import pydotplus
from sklearn.externals.six import StringIO


def getdata():
    raceratings = da.getraceratings()
    raceratings = raceratings.drop(columns=['year', 'round', 'circuitId', 'date', 'time', 'race'])
    results = da.results()
    raceresults = pd.merge(raceratings, results, how='right', on="raceId")
    crashers = raceresults[
        (raceresults["statusId"] > 1) & ((raceresults["statusId"] < 10) | (raceresults["statusId"] > 14))]
    ccount = crashers.groupby("raceId").count()
    cols = np.delete(ccount.columns.to_numpy(), 18)
    ccount = ccount.drop(columns=cols)
    ccount.columns = ["crashcount"]
    ccount = ccount.fillna(0)
    raceratings = pd.merge(raceratings, ccount, how="right", on="raceId")
    raceratings["crashcount"] = raceratings["crashcount"].fillna(0)
    stops = da.pitstops()
    racestops = pd.merge(raceratings, stops, how='left', on="raceId")
    racestopsraceavg = racestops.groupby(["raceId", "driverId"]).count().groupby("raceId").mean()
    racestopsraceavg = racestopsraceavg.drop(
        columns=['name', 'rating', 'crashcount', 'lap', 'time', 'duration', 'milliseconds'])
    racestopsraceavg.columns = ["averagestops"]
    raceratings = pd.merge(raceratings, racestopsraceavg, how="right", on="raceId")
    laptimes = da.laptimes()
    racelapratings = pd.merge(raceratings, laptimes, how="right", on="raceId")
    racelapratings = racelapratings.dropna(subset=["rating", ])
    avglaptimerace = racelapratings.groupby(["raceId", "driverId"]).mean().groupby("raceId").mean()
    avgbestlaptime = racelapratings.groupby(["raceId", "driverId"]).min().groupby("raceId").mean()
    avgdeltarace = (avglaptimerace["milliseconds"] - avgbestlaptime["milliseconds"])
    raceratings = pd.merge(raceratings, avgdeltarace, how="left", on="raceId")
    raceratings.columns = ['raceId', 'name', 'rating', 'crashcount', 'averagestops', 'delta']
    raceratings = raceratings.drop(columns=['raceId'])
    return raceratings.dropna()


def evalRFR(model, labels):
    # print("mean_absolute_error: " + str(mean_absolute_error(df[colT], df[colP])))
    # print("mean_squared_error: " + str(mean_squared_error(df[colT], df[colP])))
    # print("r2_score: " + str(r2_score(df[colT], df[colP])))
    i = 0
    for estimator in model.estimators_:
        export_graphviz(estimator,
                        out_file=str(i) + 'tree.dot',
                        feature_names=labels,
                        class_names=labels,
                        rounded=True, proportion=False,
                        precision=2, filled=True)
        i += 1


def modeltest(model):
    data = getdata()
    data = data.drop(columns=["name"])
    data_x = data.drop(columns=['rating'])
    data_y = data.drop(columns=['crashcount', 'averagestops', 'delta'])
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.5)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    testpredict = y_test
    testpredict["p_rating"] = y_predict

    evalRFR(model, data_x.columns)

    delta = []
    for i in range(0, len(testpredict)):
        delta.append(abs(testpredict["rating"].iloc[i] - testpredict["p_rating"].iloc[i]))
    return sum(delta) / len(delta)


def testmodels():
    dec = []
    for i in range(0, 10):
        dec.append(modeltest(DecisionTreeRegressor()))

    fore = []
    for i in range(0, 10):
        fore.append(modeltest(RandomForestRegressor(n_estimators=20)))

    lin = []
    for i in range(0, 10):
        lin.append(modeltest(LinearRegression(normalize=True)))

    print("dec")
    print(dec)
    print("for")
    print(fore)
    print("lin")
    print(lin)


def linreg():
    data = getdata()
    data = data.drop(columns=["name"])
    data_x = data.drop(columns=['rating'])
    data_y = data.drop(columns=['crashcount', 'averagestops', 'delta'])
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.8)
    model = LinearRegression(fit_intercept=True, normalize=True, copy_X=False, n_jobs=None).fit(x_train, y_train)
    y_predict = model.predict(x_test)
    testpredict = y_test
    testpredict["p_rating"] = y_predict
    delta = []
    for i in range(0, len(testpredict)):
        delta.append((abs(testpredict["rating"].iloc[i] - testpredict["p_rating"].iloc[i]) - 0.75) <= 0)
    print("mean_absolute_error: " + str(mean_absolute_error(testpredict["rating"], testpredict["p_rating"])))
    print("mean_squared_error: " + str(mean_squared_error(testpredict["rating"], testpredict["p_rating"])))
    print("r2_score: " + str(r2_score(testpredict["rating"], testpredict["p_rating"])))
    print("metric: " + str(delta.count(True) / len(delta)))

#
# for i in range(0, 10):
#     linreg()

# eval(modeltest(RandomForestRegressor(n_estimators=20)))
