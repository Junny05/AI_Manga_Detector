import requests
import os
import time

BASE = "https://api.mangadex.org"

def extract_title(item):
    title_dict = item["attributes"]["title"]
    if "en" in title_dict:
        return title_dict["en"]

    for alt in item["attributes"].get("altTitles", []):
        if "en" in alt:
            return alt["en"]

    if title_dict:
        return next(iter(title_dict.values()))

    return f"unknown_{item['id'][:8]}"

def get_popular_manga(limit=20):
    r = requests.get(f"{BASE}/manga", params={
        "limit": limit,
        "order[followedCount]": "desc",
        "contentRating[]": ["safe", "suggestive"],
    })
    r.raise_for_status()
    results = r.json()["data"]

    manga_list = []
    for item in results:
        manga_id = item["id"]
        title = extract_title(item)
        manga_list.append((manga_id, title))

    return manga_list


def get_cover_filenames(manga_id, limit=8):
    r = requests.get(f"{BASE}/cover", params={"manga[]": manga_id, "limit": limit})
    r.raise_for_status()
    results = r.json()["data"]
    return [item["attributes"]["fileName"] for item in results]


def download_covers_split(manga_id, title, limit=4):
    filenames = get_cover_filenames(manga_id, limit=limit)

    if len(filenames) < 2:
        raise ValueError(f"Only {len(filenames)} cover(s) found — need at least 2 to split")

    folder_name = title.lower().replace(" ", "_").replace("/", "_")
    ref_dir = f"references/{folder_name}"
    test_dir = f"test/{folder_name}"
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    saved = {"reference": [], "test": []}

    for i, filename in enumerate(filenames):
        image_url = f"https://uploads.mangadex.org/covers/{manga_id}/{filename}"
        response = requests.get(image_url)
        response.raise_for_status()

        is_last = (i == len(filenames) - 1)
        target_dir = test_dir if is_last else ref_dir
        save_path = f"{target_dir}/cover{i}.jpg"

        with open(save_path, "wb") as f:
            f.write(response.content)

        saved["test" if is_last else "reference"].append(save_path)

    return saved


popular_manga = get_popular_manga(limit=20)

succeeded = []
failed = []

for manga_id, title in popular_manga:
    try:
        saved = download_covers_split(manga_id, title, limit=8)
        print(f"{title}: {len(saved['reference'])} reference, {len(saved['test'])} test")
        succeeded.append(title)
    except Exception as e:
        print(f"Failed: {title} ({e})")
        failed.append(title)

    time.sleep(0.5)  

print(f"\nDone. {len(succeeded)} succeeded, {len(failed)} failed.")
if failed:
    print("Failed titles:", failed)