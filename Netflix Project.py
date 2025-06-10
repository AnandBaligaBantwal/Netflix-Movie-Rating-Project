import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(
    "combinedNetflixData.txt",
    header=None,
    usecols=[0, 1],
    names=["CustID", "Ratings"],
)
df

df.info()

df[df["CustID"].str.contains(":")]

movie_count = df[df["CustID"].str.contains(":")].count()[0]
movie_count

cust_count = df["CustID"].nunique()  # Number of unique customer id's from our data.
cust_count

cust_count = cust_count - movie_count
cust_count

ratings_count = len(df) - movie_count  # Total number of ratings in our given data
ratings_count

movie_id = []
for x in df["CustID"]:
    if ":" in x:
        y = x.replace(":", "")
    movie_id.append(y)

df["MovieID"] = movie_id
df

df.dropna(inplace=True)

df
df.info()

df["CustID"] = df["CustID"].astype(int)
df["MovieID"] = df["MovieID"].astype(int)

df.info()

df.duplicated().sum()

# Manual Filtration

data_cust_summary = df.groupby("CustID")["Ratings"].count()
data_cust_summary

cust_benchmark = round(data_cust_summary.quantile(0.60))
cust_benchmark

data_cust_summary[data_cust_summary < cust_benchmark]  # Rejected Customer_ID

Rejected_cust = data_cust_summary[data_cust_summary < cust_benchmark].index
Rejected_cust

# Filter out customers with less than 60% of the average ratings
df = df[~df["CustID"].isin(Rejected_cust)]

data_movie_summary = df.groupby("MovieID")["Ratings"].count()
data_movie_summary

movie_benchmark = round(data_movie_summary.quantile(0.60))
movie_benchmark

Rejected_movies = data_movie_summary[data_movie_summary < movie_benchmark].index
Rejected_movies

df = df[~df["MovieID"].isin(Rejected_movies)]
df

df_title = pd.read_csv(
    "NetflixMovieData.csv",
    header=None,
    usecols=[0, 1, 2],
    names=["MovieID", "Year", "Name"],
)
df_title

from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate

reader = Reader()

data = Dataset.load_from_df(df[["CustID", "MovieID", "Ratings"]][:1000000], reader)

model = SVD()

cross_validate(model, data, measures=["RMSE"], cv=3)

# Movie recommendation for user:
user_1331154 = df_title.copy()
user_1331154

# estimate ratings
user_1331154["Estimate Score"] = user_1331154["MovieID"].apply(
    lambda x: model.predict(1331154, x).est
)
user_1331154.sort_values(by="Estimate Score", ascending=False)

df["CustID"]

user_44937 = df_title.copy()
user_44937

user_44937["Estimate Score"] = user_44937["MovieID"].apply(
    lambda x: model.predict(44937, x).est
)
user_44937.sort_values(by="Estimate Score", ascending=False)
