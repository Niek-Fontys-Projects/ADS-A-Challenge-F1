import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt;

plt.rcdefaults()

import numpy as np
import matplotlib.pyplot as plt

import DataAccess.DataAccess as da


def makecmp(raceratings, names):
    avrr = raceratings.groupby('name').mean()
    avrr = avrr.sort_values("rating", ascending=False)
    unames = avrr.index.values
    cmp = plt.get_cmap('plasma', len(unames))
    colors = []
    for i in range(0, len(names)):
        colors.append(cmp.colors[np.where(unames == names[i])[0][0]])
    return ListedColormap(colors)


def avratcir():
    raceratings = da.getraceratings()
    avrr = raceratings.groupby('name').mean()

    labels = avrr.index.values
    y_pos = np.arange(0, len(labels) * 6, 6)
    avrr["labels"] = labels
    avrr = avrr.sort_values("rating", ascending=False)
    cmp = makecmp(raceratings, avrr.index)

    plt.barh(y_pos, avrr['rating'], align='center', alpha=0.8, height=4.5, color=cmp.colors)
    plt.yticks(y_pos, avrr["labels"], rotation='horizontal')
    plt.xlim(5, 9)
    plt.xlabel('Score')
    plt.title('Average rating (1-10) per grand prix')
    plt.tight_layout()
    #plt.show()

    plt.savefig("D:\Semester3\ADS-A\Challenge\Charts\AvgRatGP.png")


def crashrat():
    raceratings = da.getraceratings()
    results = da.results()
    raceresults = pd.merge(raceratings, results, how='right', on="raceId")
    raceresults = raceresults.dropna(subset=["rating"])
    crashers = raceresults[
        (raceresults["statusId"] > 1) & ((raceresults["statusId"] < 10) | (raceresults["statusId"] > 14))]
    ccount = crashers.groupby("raceId").count()
    rmean = raceresults.groupby("raceId").mean()
    racecrash = raceresults.drop_duplicates(subset=["raceId"])
    cols = np.delete(ccount.columns.to_numpy(), 24)
    ccount = ccount.drop(columns=cols)
    ccount.columns = ["crashcount"]
    racecrash = pd.merge(racecrash, ccount, how="left", on="raceId")
    cols = np.delete(rmean.columns.to_numpy(), 4)
    rmean = rmean.drop(columns=cols)
    rmean.columns = ["averagerating"]
    racecrash = pd.merge(racecrash, rmean, how="left", on="raceId")
    racecrash = racecrash.fillna(0)
    cmp = makecmp(raceratings, raceresults.drop_duplicates(subset=["raceId"]).name.to_numpy())
    plt.scatter(racecrash["crashcount"], racecrash["averagerating"], alpha=0.8, color=cmp.colors)
    plt.ylabel("Score")
    plt.xlabel("DNF's")
    plt.title("Rating (1-10) compared to DNF's per race")
    plt.tight_layout()
    #plt.show()

    plt.savefig("D:\Semester3\ADS-A\Challenge\Charts\CrashRatRace.png")


def stoprat():
    raceratings = da.getraceratings()
    stops = da.pitstops()
    racestops = pd.merge(raceratings, stops, how='right', on="raceId")
    racestops = racestops.dropna(subset=["rating"])
    racestopsdriverrace = racestops.groupby(["raceId", "driverId"]).count()
    racestopsraceavg = racestopsdriverrace.groupby("raceId").mean()
    nracestops = racestops.groupby("raceId").mean()
    cmap = makecmp(raceratings, racestops.drop_duplicates(subset=["raceId"]).name.to_numpy())
    plt.scatter(racestopsraceavg["stop"], nracestops["rating"], alpha=0.8, color=cmap.colors)
    plt.ylabel("Score")
    plt.xlabel("Pit stops")
    plt.title("Rating (1-10) compared to average pit stops per race")
    plt.tight_layout()
    #plt.show()

    plt.savefig("D:\Semester3\ADS-A\Challenge\Charts\StopRatingRace.png")


def avgofffastlap():
    raceratings = da.getraceratings()
    laptimes = da.laptimes()
    racelapratings = pd.merge(raceratings, laptimes, how="right", on="raceId")
    racelapratings = racelapratings.dropna(subset=["rating", ])
    avglaptimerace = racelapratings.groupby(["raceId", "driverId"]).mean().groupby("raceId").mean()
    avgbestlaptime = racelapratings.groupby(["raceId", "driverId"]).min().groupby("raceId").mean()
    avgdeltarace = (avglaptimerace["milliseconds"] - avgbestlaptime["milliseconds"])
    cmp = makecmp(raceratings, racelapratings.drop_duplicates(subset=["raceId"]).name.to_numpy())
    plt.scatter(avgdeltarace, avglaptimerace["rating"], alpha=0.8, color=cmp.colors)
    plt.ylabel("Score")
    plt.xlabel("Average ms off fastest lap")
    plt.title("Rating (1-10) compared to average delta per driver per race")
    plt.tight_layout()
    #plt.show()

    plt.savefig("D:\Semester3\ADS-A\Challenge\Charts\AvgOffFastLapBig.png")


crashrat()
