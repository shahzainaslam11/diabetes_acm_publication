import yaml
from src.data.preprocessing import DiabetesPreprocessor
from src.models.factory import get_model
from src.training.cross_validation import CrossValidator
from src.data.sampling import apply_smote, apply_smoteenn
from src.utils.seed import set_seed


def main():
    config = yaml.safe_load(open("configs/config.yaml"))
    set_seed(config["seed"])

    preprocessor = DiabetesPreprocessor()
    X, y = preprocessor.preprocess(config["data"]["path"])

    # Sampling (as per notebook experiments)
    X_smote, y_smote = apply_smote(X, y)
    X_smoteenn, y_smoteenn = apply_smoteenn(X, y)

    models = ["rf", "xgb", "lgbm"]

    for model_name in models:
        print(f"\nRunning model: {model_name}")

        model = get_model(model_name, config["models"][model_name])
        cv = CrossValidator(model, config["training"])

        print("Original Data:")
        print(cv.run(X, y))

        print("SMOTE:")
        print(cv.run(X_smote, y_smote))

        print("SMOTE-ENN:")
        print(cv.run(X_smoteenn, y_smoteenn))


if __name__ == "__main__":
    main()
