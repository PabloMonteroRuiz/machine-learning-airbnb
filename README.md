# Madrid Real Estate – Sale Price Prediction

## Authors

## Repository

## 1. Business Problem

Madrid is one of the most dynamic residential real estate markets in Spain. The dataset used in this project contains approximately 94,800 property sale listings collected quarterly throughout 2018, covering the entire municipality. In this market, accurately pricing a property is a critical decision for sellers, buyers, and investors alike. An overpriced property stays on the market for months, while an underpriced one leaves money on the table.

Despite its importance, property valuation in Spain still relies heavily on subjective appraisals. Traditional methods include comparing a handful of nearby sales, consulting real estate agents, or using the cadastral reference value — none of which systematically account for how every characteristic of a property (size, floor, amenities, building quality, distance to transport) contributes to its market price. Online valuation tools offered by portals like Idealista or Fotocasa provide rough estimates, but they are opaque, offer no explanations, and are optimised for the platform's engagement rather than the user's accuracy.

This project aims to build a transparent, data-driven model that estimates the sale price of a residential property in Madrid from its physical characteristics, location, and building attributes. Such a model can help sellers set a competitive asking price, help buyers assess whether a listing is fairly priced, assist investors in identifying undervalued properties, and give public authorities an objective tool for monitoring the housing market.

## 2. Data Source

The dataset comes from the [Idealista / Universidad Politécnica de Madrid open dataset](https://www.idealista.com) of residential sale listings in Madrid, collected quarterly in 2018 (March, June, September, December). It contains **94,815 listings and 41 variables** covering property characteristics, amenities, cadastral information, and location.

**Advantages:**
- Large sample size (~95K listings) covering the entire city of Madrid across four quarters.
- Rich feature set: physical attributes (area, rooms, floor), 17 binary amenity flags, cadastral data (construction year, building quality), and pre-computed distances to key city landmarks.
- Geographic coordinates (latitude/longitude) allow spatial analysis.
- Very few missing values in most columns — only `CONSTRUCTIONYEAR` (59%) has significant missingness, but the cadastral alternative (`CADCONSTRUCTIONYEAR`) is 100% complete.

**Disadvantages:**
- Cross-sectional snapshots: listings represent asking prices, not actual transaction prices, so they may be inflated.
- Limited to 2018 — the model cannot capture post-pandemic market shifts.
- No information about the property's condition, renovation status, or interior quality.
- Some duplicate listings (~19K repeated asset IDs) require cleaning.
- `UNITPRICE` is derived directly from `PRICE` and `CONSTRUCTEDAREA`, creating a leakage variable that must be excluded.

## 3. Problem Formulation

- **Type of learning:** Supervised regression.
- **Target variable:** `PRICE` — the asking sale price of the property in EUR. Since the distribution is strongly right-skewed (range 21,000–8,133,000 €, median 262,000 €), the model will predict `log(PRICE)` and transform back to EUR for interpretation.
- **Candidate features:**

| Feature | Type | Description |
|---|---|---|
| `CONSTRUCTEDAREA` | Continuous | Built area in m² |
| `ROOMNUMBER` | Discrete | Number of rooms |
| `BATHNUMBER` | Discrete | Number of bathrooms |
| `FLOORCLEAN` | Ordinal | Floor number (-1 = basement) |
| `FLATLOCATIONID` | Categorical | Interior (1) vs exterior (2) position |
| `CADCONSTRUCTIONYEAR` | Continuous | Year of construction (cadastral source) |
| `CADASTRALQUALITYID` | Ordinal | Building quality index (0 = best, 9 = worst) |
| `CADMAXBUILDINGFLOOR` | Discrete | Maximum floors in the building |
| `CADDWELLINGCOUNT` | Discrete | Number of dwellings in the building |
| `AMENITYID` | Ordinal | Amenity level (1, 2, 3) |
| `HASTERRACE`, `HASLIFT`, `HASAIRCONDITIONING`, ... | Binary | 17 property amenity flags |
| `BUILTTYPEID_1`, `_2`, `_3` | Binary | Building type indicators |
| `DISTANCE_TO_CITY_CENTER` | Continuous | Distance to city center (km) |
| `DISTANCE_TO_METRO` | Continuous | Distance to nearest metro station (km) |
| `DISTANCE_TO_CASTELLANA` | Continuous | Distance to Paseo de la Castellana (km) |
| `LATITUDE`, `LONGITUDE` | Continuous | Geographic coordinates |
| `PERIOD` | Categorical | Quarter of listing (201803–201812) |

- **Excluded variables (leakage or non-informative):**
  - `UNITPRICE`: directly derived from the target (`PRICE / CONSTRUCTEDAREA`).
  - `ASSETID`: unique property identifier with no predictive value.
  - `CONSTRUCTIONYEAR`: 59% missing and contains erroneous values; replaced by `CADCONSTRUCTIONYEAR`.
  - `PARKINGSPACEPRICE`: inconsistent (placeholder value 1 when no parking exists); parking information is captured by `HASPARKINGSPACE` and `ISPARKINGSPACEINCLUDEDINPRICE`.

## 4. Project Structure

```
├── data/
│   ├── Madrid_Sale.csv              # Raw dataset
│   └── processed/
│       ├── listings_00.parquet      # After type fixes (notebook 00)
│       ├── listings_01.parquet      # After missing/erroneous value treatment (notebook 01)
│       └── listings_02.parquet      # After outlier treatment (notebook 02)
├── notebooks/
│   ├── 00_data_loading_and_types.ipynb
│   ├── 01_missing_and_erroneous_values.ipynb
│   ├── 02_distributions_and_outliers.ipynb
│   └── 03_bivariate_analysis_and_conclusions.ipynb
├── src/
└── README.md
```
