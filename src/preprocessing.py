
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def build_preprocessor(X):

    # Identify numerical columns
    numerical_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    # Identify categorical columns
    categorical_columns = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    # Numerical preprocessing
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    # Categorical preprocessing
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    # Combine both pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, numerical_columns),
            ("categorical", categorical_pipeline, categorical_columns)
        ]
    )

    return preprocessor
