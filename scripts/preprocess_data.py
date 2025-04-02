from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def build_preproc_pipeline():
    #Column groups
    num_features = ['credit_score', 'age', 'tenure', 'balance', 'products_number', 'estimated_salary']
    cat_featurs = ['country', 'gender']
    pasthrough_features = ['credit_card', 'active_member']
    
    #Pipeline for numerical features
    numerical_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    #Pipeline for categorical features
    categorical_pipeline = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    #Full preprocessor pipeline
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_pipeline, num_features),
        ('cat', categorical_pipeline, cat_featurs),
        ('pass', 'passthrough', pasthrough_features)
    ])
    
    return preprocessor