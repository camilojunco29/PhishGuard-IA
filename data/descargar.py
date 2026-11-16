from hashlib import sha256
from pathlib import Path
from urllib.request import Request, urlopen

URL = ("https://data.mendeley.com/public-files/datasets/hz2d6gz7pc/files/"
       "f796c8e2-3768-4c2d-8b73-48f0d7771de5/file_downloaded")
EXPECTED = "656b2245d58da72d640680e5c2a168673a130b38607f2a427c773bbb167e995e"

with urlopen(Request(URL, headers={"User-Agent": "Mozilla/5.0"}), timeout=60) as response:
    content = response.read()
if sha256(content).hexdigest() != EXPECTED:
    raise SystemExit("El archivo descargado no coincide con SpaPhish v5.")
target = Path(__file__).with_name("spaphish.csv")
target.write_bytes(content)
print(f"SpaPhish v5 verificado en {target}")
