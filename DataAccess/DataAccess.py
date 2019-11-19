import pandas as pd


def ratings():
    ratings = pd.read_csv(r"D:\Semester3\ADS-A\Challenge\DataSets\fanRaceRatings.csv")
    return ratings.drop(columns=["P1", "P2", "P3"])


def circuits():
    circuits = pd.read_csv(r"D:\Semester3\ADS-A\Challenge\DataSets\circuits.csv", encoding="iso-8859-1")
    return circuits.drop(columns=["url", "alt"])


def races():
    races = pd.read_csv(r"D:\Semester3\ADS-A\Challenge\DataSets\races.csv", encoding="iso-8859-1")
    return races.drop(columns=["url"])


def laptimes():
    laptimes = pd.read_csv(r"D:\Semester3\ADS-A\Challenge\DataSets\lapTimes.csv", encoding="iso-8859-1")
    return laptimes


def qualifyingtimes():
    qualifying = pd.read_csv(r"D:\Semester3\ADS-A\Challenge\DataSets\qualifying.csv", encoding="iso-8859-1")
    return qualifying


def pitstops():
    pitstops = pd.read_csv(r"D:\Semester3\ADS-A\Challenge\DataSets\pitStops.csv", encoding="iso-8859-1")
    return pitstops


def drivers():
    drivers = pd.read_csv(r"D:\Semester3\ADS-A\Challenge\DataSets\drivers.csv", encoding="iso-8859-1")
    return drivers.drop(columns=["url"])


def driversstandings():
    driversstandings = pd.read_csv(r"D:\Semester3\ADS-A\Challenge\DataSets\driverStandings.csv", encoding="iso-8859-1")
    return driversstandings


def results():
    results = pd.read_csv(r"D:\Semester3\ADS-A\Challenge\DataSets\results.csv", encoding="iso-8859-1")
    return results


def statusses():
    statusses = pd.read_csv(r"D:\Semester3\ADS-A\Challenge\DataSets\status.csv", encoding="iso-8859-1")
    return statusses

def replaceGP(value):
    return value.replace("Grand Prix", "GP")



def getraceratings():
    race = races()
    rating = ratings()
    raceratings = pd.merge(race, rating, how='left', on=['year', 'name'])
    raceratings = raceratings.dropna(subset=["rating"])
    raceratings['name'] = raceratings['name'].apply(lambda x: replaceGP(x))
    return raceratings