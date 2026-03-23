from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN


def apply_smote(X, y, random_state: int = 42):
    """
    Applies SMOTE oversampling.
    """
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled


def apply_smoteenn(X, y, random_state: int = 42):
    """
    Applies SMOTE-ENN hybrid sampling
    """
    smote_enn = SMOTEENN(random_state=random_state)
    X_resampled, y_resampled = smote_enn.fit_resample(X, y)
    return X_resampled, y_resampled
