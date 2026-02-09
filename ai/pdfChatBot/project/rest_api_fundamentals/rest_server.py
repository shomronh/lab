from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel

import uvicorn

app = FastAPI(title="Simple library API")

class Book(BaseModel):
    id: Optional[int] = None
    title: str
    author: str
    pages: int

books_db = []
current_id = 1

@app.get("/books", response_model=List[Book])
def get_all_books():
    return books_db

@app.post("/books", response_model=Book, status_code=201)
def create_book(book: Book):
    global current_id
    book.id = current_id

    # generally we don't just add 1 to the id, but for simplicity 
    # we will do it here
    # in a real application we would use a database that would 
    # handle the id generation for us
    current_id += 1

    books_db.append(book)
    return book

if __name__ == "__main__":
    # uvcorn.run(app, host="0.0.0.0", port=8000)
    uvicorn.run("main:app", host="0.0.0.0", port=8000)