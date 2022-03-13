# Databricks notebook source
def regression_feature_importance(train, test, output=False):
  
  print("Step 1: train test data split")
  print("Training Data shape: ", train.shape)
  print("Training Data shape: ", train.shape,"\n")
  X_train, X_test, y_train, y_test = train_test_split(train, test, test_size = 0.2, random_state = 101) 
  
  
  print("\n")
  print("Step2: choose feature selection Regression Models")
  
  test_results={}
  model_dict = {
 'Gradient Boosted Regressor': GradientBoostingRegressor(),
  'Decision Tree Regressor':DecisionTreeRegressor(),
  'RandomForest Regressor':RandomForestRegressor(),
  'Linear Regression':LinearRegression()
  #'RuleFit Regressor':RuleFit(random_state = 101,max_rules = 500)
  }
  
  print("We apply feature importance using Regression models")
  for model_name in model_dict.keys():
    print("   ", model_name)
  
  
  print("\n")
  print("\n","Step3: Run models, display top feature importance for each model")
  importance_features_sorted_all= pd.DataFrame()
    
    
  for model_name, model in model_dict.items():
    print('-'*10, model_name, '-'*10)
    if model_name == 'RuleFit Regressor':
        model_dict[model_name].fit(X_train,y_train, feature_names = features)
    else:
        model_dict[model_name].fit(X_train,y_train)
    r2 = r2_score(model_dict[model_name].predict(X_test),y_test)
    test_results.update({model:np.round(r2,2)})

    importance_values = np.absolute(model.coef_) if model_name=='Linear Regression' else model.feature_importances_
    importance_features_sorted = pd.DataFrame(importance_values.reshape([-1,len(train.columns)]), columns=train.columns).mean(axis=0).sort_values(ascending=False).to_frame()
    importance_features_sorted.rename(columns={0:'feature_importance'},inplace=True)
    importance_features_sorted['ranking'] = importance_features_sorted['feature_importance'].rank(ascending=False)
    importance_features_sorted['model'] = model_name
    print('Show top 10 important features:')
    print(importance_features_sorted.head(10))
    importance_features_sorted_all = importance_features_sorted_all.append(importance_features_sorted)
    estimator_dict[model_name] = model
    
  print(test_results)
  if output:
    return estimator_dict

  

# COMMAND ----------

def classification_feature_importance(train, test, output=False):

  
  print("Step 1: train test data split")
  print("Training Data shape: ", train.shape)
  print("Training Data shape: ", train.shape,"\n")
  

  X_model,X_valid,y_model,y_valid = train_test_split(train,test,random_state=random_state,test_size=0.8)
  
  print("\n","Step2: choose feature selection Classification Models")
  # 1. Logistic Regression with L1 penalty
  # 2. ExtraTreesClassifier
  # 3. Random Forest

  model_dict = {'LogisticRegression':LogisticRegression(penalty='l1',solver='saga',C=2,multi_class='multinomial',n_jobs=-1,random_state=random_state),
  'ExtraTreesClassifier':ExtraTreesClassifier(n_estimators=200,max_depth=3,min_samples_leaf=0.06,n_jobs=-1,random_state=random_state),
  'RandomForestClassifier':RandomForestClassifier(n_estimators=200,max_depth=2,min_samples_leaf=0.1,random_state=random_state,n_jobs=-1)}
  
  
  print("We apply feature importance using Classification models")
  for model_name in model_dict.keys():
    print(model_name)

    
  print("\n","Step3: Run models, store top feature importance")
  estimator_dict = {}
  importance_features_sorted_all = pd.DataFrame()

  for model_name, model in model_dict.items():
    print('-'*10, model_name, '-'*10)
    model.fit(X_model,y_model)
    print('Accuracy in training:',accuracy_score(model.predict(X_model),y_model))
    print('Accuracy in valid:',accuracy_score(model.predict(X_valid),y_valid))
    importance_values = np.absolute(model.coef_) if model_name=='LogisticRegression' else model.feature_importances_
    importance_features_sorted = pd.DataFrame(importance_values.reshape([-1,len(train.columns)]), columns=train.columns).mean(axis=0).sort_values(ascending=False).to_frame()
    importance_features_sorted.rename(columns={0:'feature_importance'},inplace=True)
    importance_features_sorted['ranking'] = importance_features_sorted['feature_importance'].rank(ascending=False)
    importance_features_sorted['model'] = model_name
    print('Show top 10 important features:')
    print(importance_features_sorted.head(10))
    importance_features_sorted_all = importance_features_sorted_all.append(importance_features_sorted)
    estimator_dict[model_name] = model
  
  if output:
    return estimator_dict


# COMMAND ----------

