import requests
from bs4 import BeautifulSoup
import os
import time
from urllib.parse import urljoin

BASE_URL = "https://www.thomann.pl/modele_t.html?marketingAttributes%5B%5D=EXCLUDE_BUNDLE&oa=pra&gk=GIEGTE&sp=solr_improved&cme=true&filter=true"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "pl-PL,pl;q=0.9,en-US;q=0.8,en;q=0.7",
}

EXCLUDED_WORDS = ["cookie", "logo", "icon", "banner", "consent", "payment", "social"]

os.makedirs("data", exist_ok=True)

all_links = set()
page = 1

# -------------------------
# 1. Zbieranie linków do produktów
# -------------------------
while True:
    url = f"{BASE_URL}&page={page}"
    print(f"➡️ Analiza: Strona {page}")

    try:
        res = requests.get(url, headers=headers)
        res.raise_for_status()
        
    except requests.exceptions.RequestException as e:
        print(f"Błąd sieci przy pobieraniu strony {page}: {e}")
        break

    soup = BeautifulSoup(res.text, "html.parser")
    product_links = []

    for a in soup.select("a.product__content__title"):
        href = a.get("href")
        if href:
            full_url = urljoin("https://www.thomann.pl", href)
            full_url = full_url.split("?")[0]
            product_links.append(full_url)

    if not product_links:
        for a in soup.find_all("a", href=True):
            href = a.get("href")
            if ".htm" in href and "wszystkie-produkty" not in href:
                full_url = urljoin("https://www.thomann.pl", href).split("?")[0]
                product_links.append(full_url)

    product_links = list(set(product_links))

    if not product_links:
        print("Koniec stron lub brak linków z produktami na tej podstronie.")
        break

    all_links.update(product_links)

    # ZABEZPIECZENIE: Pobiera na razie tylko 2 strony
    if page >= 2:
        break

    time.sleep(1.5)
    page += 1

print(f"\nZnaleziono {len(all_links)} produktów. Rozpoczynam pobieranie zdjęć...\n")

# -------------------------
# 2. Pobieranie zdjęć
# -------------------------
for idx, link in enumerate(all_links):
    try:
        print(f"Pobieranie dla produktu [{idx+1}/{len(all_links)}]: {link}")

        res = requests.get(link, headers=headers)
        soup = BeautifulSoup(res.text, "html.parser")

        images = []

        meta_og = soup.find("meta", property="og:image")
        if meta_og and meta_og.get("content"):
            images.append(meta_og["content"])

        for img in soup.find_all("img"):
            src = img.get("src") or img.get("data-src")

            if src:
                src_lower = src.lower()
                
                if ("/pics/" in src_lower or "/prod/" in src_lower) and not src_lower.endswith((".png", ".svg", ".gif")):
                    if not any(bad_word in src_lower for bad_word in EXCLUDED_WORDS):
                        full_img = urljoin("https://www.thomann.pl", src)
                        if full_img not in images:
                            images.append(full_img)

        # ZMIANA TUTAJ: Pobieramy obraz z indeksu 1 (drugi) i 2 (trzeci)
        for i, img_url in enumerate(images[1:3]):
            img_res = requests.get(img_url, headers=headers)
            
            if img_res.status_code == 200:
                # 'i' będzie równe 0 dla pierwszego elementu z wycinka, więc dodajemy 2, 
                # aby pliki nazywały się _2 i _3
                with open(f"data/product_{idx}_{i+2}.jpg", "wb") as f:
                    f.write(img_res.content)
                print(f"   [+] Zapisano: product_{idx}_{i+2}.jpg")
            else:
                print(f"  ❌ Odmowa pobrania pliku: {img_url} (Kod HTTP {img_res.status_code})")

        time.sleep(1.5)

    except Exception as e:
        print(f"Błąd przy przetwarzaniu produktu {link}: {e}")

print("\nGotowe. Wszystkie pliki zostały zapisane do folderu /data.")