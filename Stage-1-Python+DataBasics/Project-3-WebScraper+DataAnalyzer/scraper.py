import requests
from bs4 import BeautifulSoup
import pandas as pd

rating_map = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}

book_list = []

for page in range(1, 51):  # pages 1 to 50
    url = f"http://books.toscrape.com/catalogue/page-{page}.html"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    books = soup.find_all("article", class_="product_pod")

    for book in books:
        title = book.h3.a["title"]
        price = book.find("p", class_="price_color").text.strip()
        price = float(price.replace("Â£", "").replace("£", ""))
        rating_class = book.find("p", class_="star-rating")["class"][1]
        rating = rating_map[rating_class]
        availability = book.find("p", class_="instock availability").text.strip()

        book_list.append(
            {
                "title": title,
                "price": price,
                "rating": rating,
                "availability": availability,
            }
        )

    print(f"Scraped page {page}/50")

df = pd.DataFrame(book_list)
print(f"\nTotal books scraped: {len(df)}")
df.to_csv("books.csv", index=False)
print("Saved to books.csv!")
