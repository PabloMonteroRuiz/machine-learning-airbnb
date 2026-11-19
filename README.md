# Madrid Real Estate – Sale Price Prediction

## Authors

- Pablo Montero Ruiz – pablo.montero@cunef.edu
- José Miguel Serrano – josemiguel.serrano@cunef.edu

## Repository

https://github.com/PabloMonteroRuiz/machine-learning-airbnb

## 1. Business Problem

Madrid is one of the most dynamic residential real estate markets in Spain. In this market, pricing a property correctly is a critical decision for sellers, buyers and investors: an overpriced property stays on the market for months, while an underpriced one leaves money on the table.

Despite its importance, property valuation in Spain still relies heavily on subjective judgement. Traditional methods include comparing a handful of nearby listings, asking real estate agents, or using the cadastral reference value. None of them systematically measures how each characteristic of a property (size, floor, building quality, amenities, location, distance to transport) contributes to its market price. Online valuation tools offered by portals such as Idealista or Fotocasa provide quick estimates, but they are black boxes: they do not disclose their methodology, do not explain which characteristics drive the estimate, and cannot be audited by the user.

This project aims to build a transparent, data-driven model that estimates the sale price of a residential property in Madrid from its physical characteristics, building attributes and location. Such a model can help sellers set a competitive asking price, help buyers assess whether a listing is fairly priced, help investors identify undervalued properties, and give public authorities an objective tool for monitoring the housing market.

## 2. Data Source

**Dataset:** Madrid sale listings from the *idealista18* open data product, downloaded from Kaggle:
https://www.kaggle.com/datasets/tomstravis/georefrenced-spain-housing (file `Madrid_Sale.csv`).

The data were originally published on idealista.com and released as an open, fully documented R package (https://github.com/paezha/idealista18, licence ODbL v1.0) for scientific and educational use. Reference article: Rey-Blanco, D., Arbués, P., López, F. A. & Páez, A. (2024). *A geo-referenced micro-data set of real estate listings for Spain's three largest cities*. Environment and Planning B: Urban Analytics and City Science. https://doi.org/10.1177/23998083241242844

The file contains **94,815 listings and 41 variables**, extracted quarterly during 2018 (March, June, September and December). Listings are enriched with official cadastral information (construction year, building quality, number of floors and dwellings) and with distances to the city centre, the nearest metro station and Paseo de la Castellana.

**Advantages:**
- Large sample covering the whole municipality of Madrid: 75,795 unique properties after removing repeated listings.
- Rich feature set: physical attributes (area, bedrooms, bathrooms, floor), 16 binary amenity flags, condition of the flat, cadastral building data and pre-computed distances.
- Geographic coordinates allow spatial analysis.
- Documented, peer-reviewed and openly licensed source, so the meaning of every variable can be checked.
- Few missing values: apart from `CONSTRUCTIONYEAR` (59%), only `FLATLOCATIONID` (6.7%) and `FLOORCLEAN` (4.1%) have gaps, and `CADCONSTRUCTIONYEAR` is a complete alternative for the construction year.

**Disadvantages:**
- Prices are **asking prices**, not transaction prices, so they may differ from the final sale price.
- Limited to **2018**: the model cannot capture later market changes (e.g. post-pandemic price growth).
- The data are **pre-filtered by the source**: price per m² ranges only from 805 to 9,998 €/m², so very cheap or very luxurious properties are not represented.
- Coordinates are **approximate** (displaced for privacy), which limits very fine-grained location analysis.
- The same flat appears several times (re-extracted each quarter, often with an updated price), so 94,815 rows correspond to ~75,800 unique properties.
- Quarters are unbalanced (Q4 accounts for 41.5% of unique listings).
- Interior quality is only captured coarsely (a 3-level condition indicator and a kitchen/furniture level).
- `UNITPRICE` is `PRICE / CONSTRUCTEDAREA`, i.e. target leakage, and must be excluded.

**How to reproduce:** download `Madrid_Sale.csv` from the Kaggle link above, place it in `data/`, and run the notebooks in order (00 → 03). The `data/` folder is not included in the repository.

## 3. Problem Formulation

- **Type of learning:** supervised learning, **regression**.
- **Target variable:** `PRICE`, the asking sale price in EUR. The distribution is strongly right-skewed (21,000 – 8,133,000 €, median 262,000 € in the raw file), so the model will predict `log(PRICE)` and transform predictions back to EUR.
- **Unit of observation:** one property (`ASSETID`). Repeated listings of the same property are collapsed into one row with its median price.
- **Candidate features:**

| Feature | Type | Description |
|---|---|---|
| `CONSTRUCTEDAREA` | Continuous | Built area in m² |
| `ROOMNUMBER` | Discrete | Number of bedrooms |
| `BATHNUMBER` | Discrete | Number of bathrooms |
| `FLOORCLEAN` | Ordinal | Floor number (-1 = basement, 0 = ground floor) |
| `FLOOR_MISSING` | Binary | 1 if the floor was not reported (created in notebook 01) |
| `FLATLOCATIONID` | Categorical | 1 = external, 2 = internal, 0 = not reported |
| `CADCONSTRUCTIONYEAR` | Continuous | Year of construction (cadastre) |
| `CADASTRALQUALITYID` | Ordinal | Cadastral building quality (lower level = higher quality and price) |
| `CADMAXBUILDINGFLOOR` | Discrete | Number of floors of the building (cadastre) |
| `CADDWELLINGCOUNT` | Discrete | Number of dwellings in the building (cadastre) |
| `AMENITYID` | Categorical | 1 = no kitchen or furniture, 2 = kitchen, 3 = kitchen and furniture |
| `HASTERRACE`, `HASLIFT`, `HASAIRCONDITIONING`, `HASPARKINGSPACE`, `HASDOORMAN`, `ISSTUDIO`, ... | Binary | 16 amenity and property-type flags |
| `BUILTTYPEID_1`, `_2`, `_3` | Binary | Condition: new development / second hand to restore / second hand in good condition (one-hot) |
| `DISTANCE_TO_CITY_CENTER` | Continuous | Distance to the city centre (km) |
| `DISTANCE_TO_METRO` | Continuous | Distance to the nearest metro station (km) |
| `DISTANCE_TO_CASTELLANA` | Continuous | Distance to Paseo de la Castellana (km) |
| `LATITUDE`, `LONGITUDE` | Continuous | Approximate geographic coordinates |
| `PERIOD` | Categorical | Quarter of extraction (201803, 201806, 201809, 201812) |

- **Excluded variables:**
  - `UNITPRICE`: target leakage (`PRICE / CONSTRUCTEDAREA`).
  - `ASSETID`: identifier, not a feature.
  - `CONSTRUCTIONYEAR`: 59% missing and erroneous values; replaced by `CADCONSTRUCTIONYEAR`.
  - `PARKINGSPACEPRICE`: placeholder value (1) in almost all listings, with or without parking.
  - `ISPARKINGSPACEINCLUDEDINPRICE`: identical to `HASPARKINGSPACE` in 100% of listings.

## 4. Exploratory Data Analysis

| Notebook | Content | Output |
|---|---|---|
| `00_data_loading_and_types.ipynb` | Loading, data types, first overview | `listings_00.parquet` (94,815 × 41) |
| `01_missing_and_erroneous_values.ipynb` | Missing values, erroneous records, duplicates, leakage | `listings_01.parquet` (75,795 × 39) |
| `02_distributions_and_outliers.ipynb` | Distributions, outlier analysis, redundant variables | `listings_02.parquet` (75,795 × 38) |
| `03_bivariate_analysis_and_conclusions.ipynb` | Relationship of each feature with log(PRICE), conclusions for modelling | — |

Main findings: size (area, bathrooms) and building quality (cadastral quality, lift, doorman) are the strongest predictors; location acts mainly as a north–south gradient; several relationships are non-linear (construction year, bedrooms); and prices per m² rose 9.2% between Q1 and Q4 2018.

## 5. Project Structure

```
├── data/                      # Not versioned: place Madrid_Sale.csv here
│   └── processed/             # Parquet files generated by the notebooks
├── notebooks/
│   ├── 00_data_loading_and_types.ipynb
│   ├── 01_missing_and_erroneous_values.ipynb
│   ├── 02_distributions_and_outliers.ipynb
│   └── 03_bivariate_analysis_and_conclusions.ipynb
├── src/
└── README.md
```
