# Madrid Real Estate – Sale Price Prediction

## Authors

- Pablo Montero Ruiz – pablo.montero@cunef.edu
- José Miguel Serrano – josemiguel.serrano@cunef.edu

## Repository

https://github.com/PabloMonteroRuiz/machine-learning-airbnb

## 1. Business Problem

Pricing a property in Madrid is tricky. Sellers want to set a competitive price without leaving money on the table, buyers want to know if a listing is fair, and investors need to spot undervalued opportunities. Right now, most people rely on rough comparisons with nearby listings, advice from agents, or the cadastral reference value — none of which take into account all the characteristics that actually affect price (size, floor, building quality, amenities, how far it is from the metro, etc.).

Portals like Idealista and Fotocasa offer automated estimates, but they don't explain how they reach their numbers, so you can't tell what's driving the price or whether the estimate makes sense for your specific case.

The goal of this project is to build a regression model that predicts the sale price of a residential property in Madrid based on its characteristics and location. We want the model to be interpretable — not just accurate, but useful for understanding what makes a property expensive or cheap.

## 2. Data Source

We use the Madrid sale listings from the idealista18 open data product, downloaded from Kaggle:
https://www.kaggle.com/datasets/tomstravis/georefrenced-spain-housing (file Madrid_Sale.csv).

The original data were published by idealista.com and released as an open R package (https://github.com/paezha/idealista18, licence ODbL v1.0). It is documented in: Rey-Blanco et al. (2024), "A geo-referenced micro-data set of real estate listings for Spain's three largest cities", Environment and Planning B.

The file has 94,815 listings and 41 variables, extracted quarterly during 2018. Each listing includes physical attributes, amenities, cadastral building data, and distances to key locations.

Advantages:
- Large dataset covering all of Madrid, with about 75,800 unique properties after cleaning.
- Rich features: area, rooms, floor, 16 amenity flags, cadastral quality, distances, coordinates.
- Documented and peer-reviewed source — every variable has a clear definition.
- Very few missing values in most columns.

Disadvantages:
- These are asking prices, not actual transaction prices, so they may be inflated.
- Only covers 2018, so it won't reflect recent market changes.
- The source pre-filters by price per m2 (805 to 9,998 EUR/m2), excluding very cheap or very luxurious properties.
- Coordinates are approximate (displaced for privacy).
- The same flat often appears multiple times with updated prices — we collapse these into one row.
- Q4 is overrepresented (41.5% of unique listings).
- UNITPRICE is PRICE / CONSTRUCTEDAREA, which is target leakage and must be excluded.

To reproduce: download Madrid_Sale.csv from the Kaggle link, place it in data/, and run the notebooks in order (00 to 03).

## 3. Problem Formulation

This is a supervised regression problem. We want to predict PRICE (asking sale price in EUR) from the property's characteristics.

Since the price distribution is very skewed (from 21k to over 8M EUR, median around 262k), we work with log(PRICE) and transform predictions back.

Each observation is one property (identified by ASSETID). Duplicates are collapsed into one row with the median price.

Features we use:

| Feature | Type | Description |
|---|---|---|
| CONSTRUCTEDAREA | Continuous | Built area in m2 |
| ROOMNUMBER | Discrete | Bedrooms |
| BATHNUMBER | Discrete | Bathrooms |
| FLOORCLEAN | Ordinal | Floor number (-1 = basement) |
| FLOOR_MISSING | Binary | 1 if floor was not reported |
| FLATLOCATIONID | Categorical | 1 = external, 2 = internal, 0 = unknown |
| CADCONSTRUCTIONYEAR | Continuous | Construction year (from cadastre) |
| CADASTRALQUALITYID | Ordinal | Building quality (lower = better) |
| CADMAXBUILDINGFLOOR | Discrete | Floors in the building |
| CADDWELLINGCOUNT | Discrete | Dwellings in the building |
| AMENITYID | Categorical | Kitchen/furniture level (1, 2, 3) |
| HAS*/IS* columns | Binary | 16 amenity and property-type flags |
| BUILTTYPEID_1, _2, _3 | Binary | New / to restore / good condition |
| DISTANCE_TO_CITY_CENTER | Continuous | km to city centre |
| DISTANCE_TO_METRO | Continuous | km to nearest metro |
| DISTANCE_TO_CASTELLANA | Continuous | km to Paseo de la Castellana |
| LATITUDE, LONGITUDE | Continuous | Approximate coordinates |
| PERIOD | Categorical | Quarter (201803 to 201812) |

Excluded: UNITPRICE (leakage), ASSETID (identifier), CONSTRUCTIONYEAR (59% missing, replaced by CADCONSTRUCTIONYEAR), PARKINGSPACEPRICE (placeholder values), ISPARKINGSPACEINCLUDEDINPRICE (identical to HASPARKINGSPACE).

## 4. Exploratory Data Analysis

| Notebook | What it does | Output |
|---|---|---|
| 00_data_loading_and_types | Load data, check types, first look | listings_00.parquet (94,815 x 41) |
| 01_missing_and_erroneous_values | Handle missing data, errors, duplicates, leakage | listings_01.parquet (75,795 x 39) |
| 02_distributions_and_outliers | Distributions, outliers, redundant variables | listings_02.parquet (75,795 x 38) |
| 03_bivariate_analysis_and_conclusions | Feature-target relationships, conclusions | — |

Main takeaways: size (area, bathrooms) and building quality (cadastral quality, lift, doorman) are the strongest predictors. Location works mainly as a north-south gradient. Some relationships are non-linear (construction year has a U-shape, bedrooms jump at 4+). Prices per m2 went up about 9% during 2018.

## 5. Project Structure

```
├── data/                      # Not versioned — place Madrid_Sale.csv here
│   └── processed/             # Parquet files generated by the notebooks
├── notebooks/
│   ├── 00_data_loading_and_types.ipynb
│   ├── 01_missing_and_erroneous_values.ipynb
│   ├── 02_distributions_and_outliers.ipynb
│   └── 03_bivariate_analysis_and_conclusions.ipynb
├── src/
└── README.md
```
