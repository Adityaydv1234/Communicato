import os

import requests

UNSPLASH_API_URL = "https://api.unsplash.com/photos/random"


def fetch_random_image(query: str | None = None) -> dict:
    access_key = os.getenv("UNSPLASH_ACCESS_KEY")
    if not access_key:
        raise RuntimeError("UNSPLASH_ACCESS_KEY is not set in .env")

    params = {"orientation": "landscape"}
    if query:
        params["query"] = query

    response = requests.get(
        UNSPLASH_API_URL,
        params=params,
        headers={"Authorization": f"Client-ID {access_key}"},
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()
    return {
        "unsplash_id": data["id"],
        "image_url": data["urls"]["regular"],
    }
