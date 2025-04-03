from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from feature_engineering import FeatureEngineeringTransformer

def build_preproc_pipeline():
    #Column groups
    num_features = [
        'credit_score', 'age', 
        'tenure', 'balance', 
        'products_number', 'estimated_salary'
    ]
    
    cat_featurs = [
        'country', 'gender', 'age_category'
    ]
    
    pasthrough_features = [
        'credit_card', 'active_member',
        'is_old_inactive', 'is_young_and_active',
        'is_high_balance_short_tenure', 'is_low_product_and_inactive',
        'is_german_customer'    
    ]
    
    #Pipeline for numerical features
    numerical_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    #Pipeline for categorical features
    categorical_pipeline = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    #Preprocessors combined
    column_transformer = ColumnTransformer(transformers=[
        ('num', numerical_pipeline, num_features),
        ('cat', categorical_pipeline, cat_featurs),
        ('passthrough', 'passthrough', pasthrough_features)
    ])
    
    #Full pipeline including feature engineering(first) 
    #and preprocessing
    
    full_pipeline = Pipeline(steps=[
        ('feature_engineering', FeatureEngineeringTransformer()),
        ('preprocessor', column_transformer)
    ])
    
    
    return full_pipeline