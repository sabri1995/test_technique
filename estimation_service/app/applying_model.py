# QUESTIONS : 
    # Développement d'un service capable d'appliquer un modèle entrainer à une base de test (les informations nécessaires doivent être comme des paramètres des fonnctions).
    # Transformation de ce service en API (port 6000)

# NB : 
    # Vous pouvez ajouter d'autres fonctions si vous juger cela nécessaire. 
    # NE PAS METTRE DES FONCTIONS HORS LE CONTEXTE DE L'APPLICATION DU MODELE'.

# ATTENTION : Les 2 fonctions que j'ai listé ici doivent être présentes dans votre code sous les même noms
from deserializer import *
from model_training import *
import pickle
from datetime import datetime
import os

def prepare_testData(test_set):
    """Cleaning, splitting the testing data and apply the same training dumification to this data."""
    check_json(test_set)
    test_df = deserialize_json(test_set)
    x_test, y_test = clean_trainData(test_df)
    return x_test, y_test


def newest(path):
    """ a function that returns the most recent model saved"""
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    return max(paths, key=os.path.getctime)

def apply_latestStatModel(x_test, y_test):
    """Application of the last statistical model saved to the test base"""
    filename = newest('models')
    loaded_model = pickle.load(open(filename, 'rb'))
    print(newest('models'))
    y_pred = loaded_model.predict(x_test)
    
    return y_pred


