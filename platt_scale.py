import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from betacal import _BetaCal
import matplotlib.pyplot as plt

def platt_rescale(predicted_probs_train: np.ndarray[np.float64],
                true_labels_train: np.ndarray[np.signedinteger],
                predicted_probs_test: np.ndarray[np.float64],
                true_labels_test: np.ndarray[np.signedinteger]):
    # Convert probabilities to log odds if necessary
    eps = 1e-12
    predicted_probs_train = np.clip(predicted_probs_train, eps, 1 - eps)
    predicted_probs_train_log_odds = np.log(predicted_probs_train / (1 - predicted_probs_train))
    predicted_probs_test = np.clip(predicted_probs_test, eps, 1 - eps)
    predicted_probs_test_log_odds = np.log(predicted_probs_test / (1 - predicted_probs_test))
    
    # Fit a logistic regression model on the training data in log odds space
    logistic_model = LogisticRegression(fit_intercept=True)
    logistic_model.fit(predicted_probs_train_log_odds.reshape(-1, 1), true_labels_train)
    # Print model parameters
    print("Logistic Regression Model Parameters:")
    print(f"Coefficient: {logistic_model.coef_[0][0]:.4f}")
    print(f"Intercept: {logistic_model.intercept_[0]:.4f}")
    
    # Apply the learned model to scale the test probabilities in log odds space
    log_odds_scaled = (predicted_probs_test_log_odds * logistic_model.coef_[0]).reshape(-1, 1)
    if logistic_model.intercept_ is not None:
        log_odds_scaled += logistic_model.intercept_
    scaled_probs_test = 1 / (1 + np.exp(-log_odds_scaled))
    
    # Visualize the transformation
    # plt.figure(figsize=(10, 6))
    # x = np.linspace(0, 1, 100)
    # x_log_odds = np.log(x / (1 - x))
    # y_log_odds = x_log_odds * logistic_model.coef_[0] + logistic_model.intercept_
    # y = 1 / (1 + np.exp(-y_log_odds))
    # plt.plot(x, y.flatten(), '-r', label='Platt scaling fit')
    # plt.scatter(predicted_probs_train, true_labels_train, alpha=0.5, label='Training data')
    # plt.xlabel('Input probabilities')
    # plt.ylabel('Calibrated probabilities')
    # plt.legend()
    # plt.title('Platt Scaling Calibration')
    # plt.savefig('platt_scaling_calibration.png')
    # plt.close()
    
    return scaled_probs_test.flatten()

def platt_rescale_no_log_odds(predicted_probs_train: np.ndarray[np.float64],
                true_labels_train: np.ndarray[np.signedinteger],
                predicted_probs_test: np.ndarray[np.float64],
                true_labels_test: np.ndarray[np.signedinteger]):
    # Clip probabilities to avoid numerical issues
    eps = 1e-12
    predicted_probs_train = np.clip(predicted_probs_train, eps, 1 - eps)
    predicted_probs_test = np.clip(predicted_probs_test, eps, 1 - eps)
    
    # Fit a logistic regression model directly on the probabilities
    logistic_model = LogisticRegression(fit_intercept=True)
    logistic_model.fit(predicted_probs_train.reshape(-1, 1), true_labels_train)
    
    # Print model parameters
    print("Logistic Regression Model Parameters:")
    print(f"Coefficient: {logistic_model.coef_[0][0]:.4f}")
    print(f"Intercept: {logistic_model.intercept_[0]:.4f}")
    
    # Apply the learned model to scale the test probabilities
    scaled_probs_test = logistic_model.predict_proba(predicted_probs_test.reshape(-1, 1))[:, 1]
    
    # Visualize the transformation
    plt.figure(figsize=(10, 6))
    x = np.linspace(0, 1, 100)
    y = logistic_model.predict_proba(x.reshape(-1, 1))[:, 1]
    plt.plot(x, y, '-r', label='Platt scaling (no log odds) fit')
    plt.scatter(predicted_probs_train, true_labels_train, alpha=0.5, label='Training data')
    plt.xlabel('Input probabilities')
    plt.ylabel('Calibrated probabilities')
    plt.legend()
    plt.title('Platt Scaling Calibration (No Log Odds)')
    plt.savefig('platt_scaling_no_log_odds_calibration.png')
    plt.close()
    
    return scaled_probs_test

def isotonic_rescale(predicted_probs_train: np.ndarray[np.float64],
                true_labels_train: np.ndarray[np.signedinteger],
                predicted_probs_test: np.ndarray[np.float64],
                true_labels_test: np.ndarray[np.signedinteger]):
    # Fit an isotonic regression model on the training data
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(predicted_probs_train, true_labels_train)
    # Print isotonic regression details
    print("Isotonic Regression Details:")
    print(f"Number of breakpoints: {len(iso_reg.X_thresholds_)}")
    print(f"First few breakpoints: {iso_reg.X_thresholds_[:5]}")
    print(f"First few output values: {iso_reg.y_thresholds_[:5]}")
    
    # Optionally, you can visualize the transformation
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(iso_reg.X_thresholds_, iso_reg.y_thresholds_, '-r', label='Isotonic fit')
    plt.scatter(predicted_probs_train, true_labels_train, alpha=0.5, label='Training data')
    plt.xlabel('Input probabilities')
    plt.ylabel('Calibrated probabilities')
    plt.legend()
    plt.title('Isotonic Regression Calibration')
    plt.savefig('isotonic_regression_calibration.png')
    plt.close()

    # Apply the learned model to scale the test probabilities
    scaled_probs_test = iso_reg.predict(predicted_probs_test)

    return scaled_probs_test

def beta_rescale(predicted_probs_train: np.ndarray[np.float64],
                true_labels_train: np.ndarray[np.signedinteger],
                predicted_probs_test: np.ndarray[np.float64],
                true_labels_test: np.ndarray[np.signedinteger]):
    # Fit a beta calibration model on the training data
    beta_cal = _BetaCal()
    beta_cal.fit(predicted_probs_train, true_labels_train)
    
    # Print beta calibration details
    print("Beta Calibration Details:")
    print(f"Parameters: {beta_cal.map_}")
    # Optionally, you can visualize the transformation
    import matplotlib.pyplot as plt
    import numpy as np
    plt.figure(figsize=(10, 6))
    x = np.linspace(0, 1, 100)
    y = beta_cal.predict(x.reshape(-1, 1))
    plt.plot(x, y, '-r', label='Beta calibration fit')
    plt.scatter(predicted_probs_train, true_labels_train, alpha=0.5, label='Training data')
    plt.xlabel('Input probabilities')
    plt.ylabel('Calibrated probabilities')
    plt.legend()
    plt.title('Beta Calibration')
    plt.savefig('beta_calibration.png')
    plt.close()

    

    # Apply the learned model to scale the test probabilities
    scaled_probs_test = beta_cal.predict(predicted_probs_test.reshape(-1, 1))

    return scaled_probs_test

