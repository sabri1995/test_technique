# QUESTIONS :
# Développement d'un service capable de vérifier si le json envoyé par l'utilisateur est correct ou pas PUIS le convertir en dataFrame.
# Transformation de ce service en API (port 6000)

# NB : 
# Vous pouvez ajouter d'autres fonctions si vous juger cela nécessaire.
# NE PAS METTRE DES FONCTIONS HORS LE CONTEXTE DE CONVERSION DU JSON ICI.

# INDICATIONS : 
# Checker si tous les objets du json ont bien un ID
# Checker si tous les sousèobjets du json ont bien le même nombre de variables
# ...

# ATTENTION : Les 2 fonctions que j'ai listé ici doivent être présentes dans votre code sous les même noms
import pandas as pd
import json
from pandas import json_normalize
import numpy as np

def check_json(dataset):
    """Check if the sent json is correct"""
    check = True
    for i in range(len(dataset)):
        try:
            assert 'id' in (dataset[i]).keys()
            assert dataset[i].get('id') != None
        except  AssertionError:
            print("dataset error in id : line", i)
            check = False
    for i in range(len(dataset) - 1):
        try:
            assert dataset[i].get('asset_infos').keys() == dataset[i + 1].get('asset_infos').keys()
            assert dataset[i].get('asset_scores').keys() == dataset[i + 1].get('asset_scores').keys()
        except AssertionError:
            print('les variables ne sont pas les memes ')
            check = False
    return check
    


def deserialize_json(dataset):
    """Deserialize the input json to build the dataFrame to put in the statistical model"""
    df = pd.DataFrame()
    for i in range(len(dataset)):
        df1 = json_normalize(dataset[i]['asset_infos'])
        df2 = json_normalize(dataset[i]['asset_scores'])

        result = pd.concat([df1, df2], axis=1)
        result['id'] = dataset[i].get('id')
        df = pd.concat([df, result], axis=0, ignore_index=True)

    df = df.set_index('id')
    return (df)
