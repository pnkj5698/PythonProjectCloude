import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("books.csv")

# Basic stats
print("Total books:", len(df))
print("\nAverage price:", df["price"].mean().round(2))
print("Cheapest book:", df["price"].min())
print("Most expensive book:", df["price"].max())
print("\nRating distribution:")
print(df["rating"].value_counts().sort_index())

# Chart 1 - Rating distribution
plt.figure(figsize=(8, 5))
df["rating"].value_counts().sort_index().plot(kind="bar", color="steelblue")
plt.title("Number of Books by Rating")
plt.xlabel("Rating")
plt.ylabel("Number of Books")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("rating_distribution.png")
plt.show()

# Chart 2 - Price distribution
plt.figure(figsize=(8, 5))
df["price"].plot(kind="hist", bins=20, color="salmon")
plt.title("Price Distribution of Books")
plt.xlabel("Price (£)")
plt.ylabel("Number of Books")
plt.tight_layout()
plt.savefig("price_distribution.png")
plt.show()
