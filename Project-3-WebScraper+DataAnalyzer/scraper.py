import requests
from bs4 import BeautifulSoup
import pandas as pd


url = "http://books.toscrape.com"
response = requests.get(url)

# print(response.status_code)

# Parse the HTML
soup = BeautifulSoup(response.text, "html.parser")

# Find all books on the page
books = soup.find_all("article", class_="product_pod")

print(f"Found {len(books)} books on this page")
# print(books[0])  # print first book's raw HTML

# Rating words to numbers
rating_map = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}

book_list = []

for book in books:
    # Extract title
    title = book.h3.a["title"]

    # Extract price
    price = book.find("p", class_="price_color").text.strip()
    price = float(price.replace("Â£", "").replace("£", ""))

    # Extract rating
    rating_class = book.find("p", class_="star-rating")["class"][1]
    rating = rating_map[rating_class]

    # Extract availability
    availability = book.find("p", class_="instock availability").text.strip()

    book_list.append(
        {"title": title, "price": price, "rating": rating, "availability": availability}
    )

# Convert to dataframe
df = pd.DataFrame(book_list)
print(df)
