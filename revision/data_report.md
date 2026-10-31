# Data report

Source `data/FiresRu.csv`: 26681 rows, 9 columns: `type_name`, `type_id`, `lon`, `lat`, `temperature_c`, `precipitation_mm`, `relative_humidity`, `wind_speed_ms`, `solar_radiation`

## Column roles
- temp: `temperature_c`
- rh: `relative_humidity`
- wind: `wind_speed_ms`
- precip: `precipitation_mm`
- solar: `solar_radiation`
- lat: `lat`
- lon: `lon`
- label: `type_name`

## Raw label values
- Forest fire: 18415
- Natural fire: 5195
- Uncontrolled burn: 2711
- Controlled burn: 333
- Peat fire: 27

type_id x label:
```
type_id  type_name        
1        Uncontrolled burn     2711
2        Peat fire               27
3        Forest fire          18415
4        Natural fire          5195
5        Controlled burn        333
```

## Filtering
- raw records: 26681
- merge rule: {'Forest fire': 'Wildfire', 'Natural fire': 'Wildfire', 'Peat fire': 'Wildfire'}; ordinal order: ['Controlled burn', 'Uncontrolled burn', 'Wildfire']
- label in order (after merge): 26681
- ... with complete drivers and coordinates: 26681
- final N after min-class filter: 26681

## Classes
- Controlled burn: 333 (0.0125)
- Uncontrolled burn: 2711 (0.1016)
- Wildfire: 23637 (0.8859)
- majority-class macro-F1: 0.3132

## Coordinates
- unique coordinate pairs (4 dp): 26680 for 26681 records
- records at a coordinate shared with others: 0.0001
- max records per coordinate: 2
- lat [41.706, 70.333], lon [20.372, 175.869]

## Meteorological join (records with identical weather tuples)
- variables: ['temp', 'rh', 'wind', 'precip', 'solar']
- distinct weather tuples: 10513 for 26681 records
- records sharing their tuple with others: 0.8046
- median / max records per tuple: 2 / 36
- tuples with >1 record: 5300; median lat span 0.033 deg, median lon span 0.058 deg, max lat span 0.493 deg
- coordinates with >1 record: 1; share of them with a single weather tuple: 0.0000
- Reading: if repeated records at one coordinate share one tuple and tuples cover
  a fixed lat/lon cell, the weather is a gridded or static join rather than a
  time-of-detection join. State the source, resolution and matching rule.

## Spatial blocks
| deg | blocks | blocks with rarest class | median records/block | largest block share |
|---|---|---|---|---|
| 1 | 1080 | 116 | 7 | 0.017 |
| 2 | 429 | 78 | 12 | 0.047 |
| 5 | 115 | 48 | 24 | 0.103 |
| 10 | 40 | 24 | 119 | 0.199 |

## Variables (check units)
```
          count    mean     std    min    25%    50%    75%    max
temp    26681.0  17.887   5.681 -34.88  15.70  19.08  21.39  31.94
rh      26681.0  69.444  11.433  24.70  62.93  71.35  77.91  99.83
wind    26681.0   0.631   0.925   0.00   0.11   0.25   0.66   7.42
precip  26681.0   0.458   1.458   0.00   0.00   0.04   0.30  25.85
solar   26681.0   5.573   1.504   0.42   4.55   5.68   6.71   8.97
```
- share of exact zeros: temp=0.000, rh=0.000, wind=0.043, precip=0.314, solar=0.000

## Exogeneity screen of auxiliary candidates (full data)
```
feature  r2_on_drivers  resid_rel_std  exogenous
  solar       0.264586       0.857563       True
```

## Time information
- No date/time column in the file: temporal and spatio-temporal protocols are
  unavailable and seasonality cannot be controlled. The paper must explain how
  weather was matched 'at the time of detection' (e.g. joined upstream, date dropped).
