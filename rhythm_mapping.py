import joblib
import numpy as np
from sklearn.exceptions import NotFittedError

try:
    KMEANS_MODEL = joblib.load('rhythm_kmeans_model.pkl')
    SCALER = joblib.load('rhythm_scaler.pkl')
except FileNotFoundError:
    print("Error: ML artifacts (rhythm_kmeans_model.pkl or rhythm_scaler.pkl) not found. Please run unsupervised_analysis.py first.")
    KMEANS_MODEL = None
    SCALER = None

# mapping clusters
CLUSTER_TO_STATE = {
    0: "Burnout State",         
    1: "Flow State",       
    2: "Balanced State", 
    3: "Fatigue State",       
    4: "Digital Drift State",     
}


# classify clusters
def classify_rhythm_state(sleep_hours, screen_time, productivity_score):
    """
    Uses the loaded K-Means model and scaler to classify a user's rhythm state.
    
    The feature order must match the training data: ['screen_time_hours', 'sleep_hours', 'productivity_0_100']
    """
    if KMEANS_MODEL is None or SCALER is None:
        return "Classification Error"

    try:
        # input order
        user_input = np.array([[screen_time, sleep_hours, productivity_score]])
        scaled_input = SCALER.transform(user_input)
        cluster_index = KMEANS_MODEL.predict(scaled_input)[0]
        return CLUSTER_TO_STATE.get(cluster_index, "Unknown State")
        
    except NotFittedError:
        return "Model Not Fitted Error"
    except Exception as e:
        return f"Classification Error: {e}"