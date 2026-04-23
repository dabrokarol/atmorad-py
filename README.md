# Symulacja transferu radiacji metodą Monte Carlo - Karol Dąbrowski

## Opis
Kod symulacji transferu radiacji w 3D przy użyciu wektoryzowanego Monte Carlo.


## Cechy programu, założenia fizyczne:
- Orientacja fotonu to Jednostkowy wektor kierunkowy
- Program symuluje równolegle strumień fotonów
- Stream Compaction - na koniec iteracji pętli usuwane są fotony, które zakończyły symulację

Upraszczające założenia:
- Jednorodna funkcja fazowa rozproszenia
- Izotropowość atmosfery (w tym brak chmur, jednorodna gęstość grubości optycznej)
- Powierzchnia ziemi pochłaniająca całość promieniowania

## Struktura projektu
```
.
├── monte_carlo.py
└── README.md
```


## Instrukcja Uruchomienia

### Pakiet UV (najprościej)

- Pobierz uv (https://docs.astral.sh/uv/getting-started/installation/)
- W folderze projektu: 
    - `uv venv`
    - `uv pip install -r requirements.txt`
    - `uv run monte_carlo.py`


## Literatura i Referencje

- Kod powstał na podstawie przedmiotu "Radiative processes in the atmosphere" odbywającego się w roku 2026 na Wydziale Fizyki Uniwersytetu Warszawskiego. Prowadził Prof. dr. hab. Krzysztof Markowicz
