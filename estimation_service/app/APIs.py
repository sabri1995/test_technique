# POST APIs

    # Api qui prend en entrée le json et qui retourne response = True si le json est correct
    # Api qui prend en entrée le json et qui retourne le max d'informations sur le modèle stocké (RMSE, R_square,...)
    # Api qui prend en entrée un json de test (même format que le json de train) et qui retourne la prédiction de la base de test

from fastapi import FastAPI

import json

app = FastAPI()

json_data = json.load(open('data/train_input_cv.json'))

@app.get("/")
def get_data():
    return json_data

