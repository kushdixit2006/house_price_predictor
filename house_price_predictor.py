import numpy as np
import pandas as pd
from sklearn.model_selection import KFold,cross_val_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error


#**************************** load data *********************************#
test_df=pd.read_csv("house-prices-advanced-regression-techniques/test.csv")
train_df = pd.read_csv("house-prices-advanced-regression-techniques/train.csv")
print(train_df.head())
print("shape of data:",train_df.shape)
print(train_df.isnull().sum().head(5))



#***************************** separate features***************************#
X=train_df.drop("SalePrice",axis=1)
y=train_df["SalePrice"]
y=np.log1p(y)



#***************handle and fill  missing values ****************************#
# check missing values
missing=X.isnull().sum()
print(missing[missing>0].sort_values(ascending=False))
# Numeric columns: fill with median
num_cols = X.select_dtypes(include=['int64','float64']).columns
# Categorical columns: fill with mode (need [0], otherwise it fails!)
cat_cols = X.select_dtypes(include=['object']).columns
# Verify no missing values
print("Total missing values:", X.isnull().sum().sum())
print(X.shape)


#************************** Preprocess data ********************************#
num_pipeline=Pipeline(steps=[
    ("imputer",SimpleImputer(strategy="median")),
    ("scaler",StandardScaler())
])
cat_pipeline=Pipeline(steps=[
    ("imputer",SimpleImputer(strategy="most_frequent")),
    ("encoder",OneHotEncoder(handle_unknown="ignore"))
])
preprocessor=ColumnTransformer(transformers=[
    ("num",num_pipeline,num_cols),
    ("cat",cat_pipeline,cat_cols)
])
model_pipeline=Pipeline(steps=[
    ("preprocessor",preprocessor),
    ("model",Ridge(alpha=1.0))
])
ridge_pipeline=Pipeline(steps=[
    ("preprocessor",preprocessor),
    ("model",Ridge(alpha=1.0))

])

#************************** cross validation ********************************#
kf=KFold(n_splits=10,shuffle=True,random_state=42)
alphas=[0.01,0.1,1,10,50,100]
for alpha in alphas:
    ridge_pipeline.set_params(model__alpha=alpha)
    scores=cross_val_score(ridge_pipeline,X,y,scoring='neg_mean_squared_error',cv=kf)
    rmse=np.sqrt(-scores.mean())
    print(f"Alpha:{alpha},CV RMSE:{rmse:.2f}")

#************************** train final model ********************************#
best_alpha=10
ridge_pipeline.set_params(model__alpha=best_alpha)
ridge_pipeline.fit(X,y)

#************************* prediction on test data ***************************#
X_test=test_df.copy()
predictions=ridge_pipeline.predict(X_test)

#******************************* submision *************************************#

submission=pd.DataFrame({
    "Id": test_df["Id"],
    "SalePrice":np.expm1(predictions)
})
submission.to_csv("submission.csv",index=False)

print(pd.read_csv("submission.csv").head(10))