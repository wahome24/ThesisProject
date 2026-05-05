#Package Import
import pandas as pd

#Data Paths
input_path = "../Data/News/news_data.csv"
output_path = "../Data/News/sampled_news_data.csv"

#Loading the data
df = pd.read_csv(input_path,low_memory=False)

#Creating a year column from date
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Year'] = df['Date'].dt.year
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

# This removes 2009-2018 data and 2024+ data
df = df[(df['Year'] >= 2019) & (df['Year'] <= 2023)].copy()

#Dropping any rows that failed the year extraction
df = df.dropna(subset=['Year'])
df['Year'] = df['Year'].astype(int)

#Random Sampling
#Grouping is done by symbol and year, then 40 random samples taken.
#random_state=42 ensures reproducibility.
balanced_df = df.groupby(['Stock_symbol', 'Year']).apply(
    lambda x: x.sample(n=min(len(x), 40), random_state=42)
).reset_index(drop=True)

#Saving the data
balanced_df.to_csv(output_path, index=False)

