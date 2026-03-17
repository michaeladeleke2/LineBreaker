# LineBreaker 🏀

**Beat the Line. Break the Line.**

ML-powered NBA prop predictions. 16 stat targets, XGBoost models trained on 5 seasons of data, live injury/lineup integration.

## Setup

```bash
pip install -r requirements.txt
python data/fetch_data.py        # fetch game logs
python features/engineer.py     # build feature matrix  
python models/train.py          # train models
streamlit run app.py            # launch app
```

## Maintenance

```bash
python retrain.py               # monthly retrain
python models/backtest.py --target pts --days 30  # check accuracy
```

## Stack

- **Data**: NBA API, ESPN (injuries + lineups)
- **Models**: XGBoost (regressor + classifier per target)
- **UI**: Streamlit
