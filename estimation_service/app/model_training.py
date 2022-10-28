# QUESTIONS : 
    # Développement d'un service capable d'entrainer un modèle statistique de prédiction sur la base de train (les informations nécessaires doivent être comme des paramètres des fonnctions).
    # Transformation de ce service en API (port 6000)

# NB : 
    # Vous pouvez ajouter d'autres fonctions si vous juger cela nécessaire. 
    # NE PAS METTRE DES FONCTIONS HORS LE CONTEXTE DE LA PREDICTION.

# ATTENTION : Les 4 fonctions que j'ai listé ici doivent être présentes dans votre code sous les même noms
from deserializer import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle
from datetime import datetime
from sklearn.model_selection import GridSearchCV


def clean_trainData(df):
    """Cleaning, splitting the training data and dummies the cateogial variables"""
    #remove duplicate rows if exists
    df.drop_duplicates()
    # catch missing values
    for column in df:
        try:
            assert df[column].isna().sum() < 1
        except AssertionError:
            print('missing value, we have to fill NaN value')
    #dummies the cateogial variables
    df['var4'] = df['var4'].map({'mod2': 2, 'mod1': 1, 'mod3': 3})
    df['var6'] = df['var6'].map({'Inner Rim_East': 0, 'Inner Rim_North': 1, 'Inner Rim_South': 2,
                                 'La Defense': 3, 'Neuilly_Levallois': 4, 'Paris 12_13':5, 'Paris 14_15':6, 'Paris 3_4_10_11': 7,
                                 'Paris 5_6_7': 8, 'Paris CBD':9, 'Paris Ouest_Hors QCA': 10, 'Peri_Defense':11,
                                 'Southern River Bend': 12 })
    df['var8'] = df['var8'].map({'Multi var8': 0, 'Single var8': 1, 'Vacant': 2})
    df['var9'] = df['var9'].map({'NO': 0, 'YES': 1})
    #split data
    
    y_data = df.target
    x_data = df.drop('target',axis=1)
    return x_data, y_data


def train_model(x_train, y_train):
    """Apply the model to the train data and the train target by turning multiple parameters"""
    '''y = y_train.values
    X = x_train.values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)'''
    param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
    }
    rf = RandomForestRegressor()
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)
    grid_search.fit(x_train, y_train)
    best_grid = grid_search.best_estimator_     
    #model = forest.fit(x_train, y_train)
    return best_grid


def get_parameters(x_test, y_test, model):
    """Calculate statistical parameters of the model (EX : RMSE)"""
    predictions = model.predict(x_test)
    MAE= mean_absolute_error(y_test,predictions)
    MSE=mean_squared_error(y_test,predictions)
    return  MAE, MSE
    

def save_model(model):
    
    model_name= str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')) + '.sav'
    pickle.dump(model, open('models/'+model_name, 'wb'))

    
