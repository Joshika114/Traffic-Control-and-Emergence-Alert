import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA

def engineer_features(df):
    df['total_vehicles'] = df['lv'] + df['2_wheeler'] + df['hv']
    df['hv_ratio'] = df['hv'] / df['total_vehicles']
    df['lv_2w_ratio'] = df['lv'] / (df['2_wheeler'] + 1)
    
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(df[['lv', '2_wheeler', 'hv']])
    poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(['lv', '2_wheeler', 'hv']), index=df.index)
    
    df = pd.concat([df, poly_df.drop(columns=['lv', '2_wheeler', 'hv'])], axis=1)
    
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(poly_df.drop(columns=['lv', '2_wheeler', 'hv']))
    df['pca_1'], df['pca_2'] = pca_features[:, 0], pca_features[:, 1]
    
    return df
