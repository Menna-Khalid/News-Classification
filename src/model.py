import pickle
def load_model(model_path):
    """Load a model from a file."""
    import pickle
    with open(model_path, 'rb') as file:
        return pickle.load(file)
def save_model(model, model_path):
    """Save a trained model."""
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {model_path}")

def predict(model, X):
    """Predict the class for a given input."""
    return model.predict(X)[0]


