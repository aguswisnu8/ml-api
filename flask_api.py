# -----------------------------------------For ML
from tensorflow.keras.models import load_model
import numpy as np 
import pandas as pd 
import json
# -----------------------------------------Flask API
from flask import Flask, jsonify, request


# -----------------------------------------ML Proses
model = load_model('model1.h5')
dtw = pd.read_csv('torem_dtw.csv')
rating = pd.read_csv('ratings.csv')

def make_pred(user_id, dtw_id, model):
    return model.predict([np.array([user_id]), np.array([dtw_id])])[0][0]

def get_topN_rec(user_id, model):
    
    user_id = int(user_id) - 1
    user_ratings = rating[rating['user_id'] == user_id]
    recommendation = rating[~rating['dtw_id'].isin(user_ratings['dtw_id'])][['dtw_id']].drop_duplicates()
    recommendation['rating_predict'] = recommendation.apply(lambda x: make_pred(user_id, x['dtw_id'], model), axis=1)
    
    final_rec = recommendation.sort_values(by='rating_predict', ascending=False).merge(dtw[['dtw_id', 'name','gmap_id']], on='dtw_id').head(10)
    
    return final_rec.sort_values('rating_predict', ascending=False)[['name','gmap_id', 'rating_predict']]
# -----------------------------------------ML Output
# a = get_topN_rec(190, model)
# print(type(a), a.shape)
# print(a)

# -----------------------------------------Similarity
indices = pd.read_csv('indices.csv')
def print_similar_dtw(id):
    # print('Tempat yang mirip',dtw.at[id,'Nama Daya Tarik Wisata'], ':')
    id=id-1
    list_dtw = {}
    for i in indices.iloc[id][1:]:
        list_dtw[dtw.at[i,'name']]=dtw.at[i,'gmap_id']
        # print(dtw.at[id,'name'])
        # return dtw.at[i,'name']
    return list_dtw


# -----------------------------------------Flask API

app = Flask(__name__)
@app.route('/')
def home():
    return "Hello to Torem API for ML"

@app.route('/predictUser/<int:id>')
def get_pred_by_user_id(id):
    pred = get_topN_rec(id, model)
    if not pred.empty:
        return pred.to_json(orient="records")
    return jsonify({'message': 'No prediction return'})

@app.route('/similarDTW/<int:id>')
def get_simil_by_dtw_id(id):
    sim = print_similar_dtw(id)
    if sim:
        return json.dumps(sim)
    return jsonify({'message': 'No similar places return'})



app.run(port=5000)
