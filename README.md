# California-House-Price-Predictor
ML project to predict California House price. 
Seaborn and matplotlib.pyplot used to identify potential data improvments (see data_review.py).
Pandas and numpy used to modify data/ create features.
Scikit-learn and joblib to create model.
Streamlit to create an app.

Model accuracy with RAW data (see model_raw/train_model_raw.py):
RMSE 0.7456
R2: 0.5758

Model accuracy after improvments:
RMSE: 0.5609
R2: 0.6528

### Demo
![California-House-demo](media/demo.gif)
### To open app locally:
```bash
pip install -r requirements.txt
streamlit run app.py
```
