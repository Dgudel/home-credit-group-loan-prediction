# +
import numpy as np 
import pandas as pd
import random

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

from scipy import stats
from scipy.stats import chi2_contingency, norm 
import researchpy as rp

from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.proportion import proportion_confint
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import optuna

import sklearn
import sklearn.ensemble
import sklearn.model_selection
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder, FunctionTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, \
cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, f1_score, precision_score, \
recall_score, confusion_matrix, make_scorer, classification_report, mean_absolute_error,\
mean_squared_error,mean_squared_log_error,r2_score


from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression, SGDClassifier
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier
import xgboost as xgb

from imblearn.pipeline import make_pipeline, Pipeline as imbPipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE

from yellowbrick.classifier import confusion_matrix, ClassificationReport



import warnings
warnings.filterwarnings('ignore')
# %matplotlib inline

# -

def score_model(x_train, y_train, x_val, y_val, estimator, preprocessor, df, 
                models_list, classes, size, features, **kwargs):
    
    # Model fitting and prediction:
    pipeline = imbPipeline(steps = [
                   ('preprocessor', preprocessor),
                    ('selectKBest', SelectKBest(score_func=mutual_info_classif, k=features)),
                #   ('rfecv', RFECV(estimator=estimator, step=1, cv=5)),
                   ('randomundersampler', RandomUnderSampler(random_state=42)),
                   ('classifier',estimator)])
    
    model = pipeline.fit(x_train, y_train)
    prediction = model.predict(x_val)
    prediction_prob = model.predict_proba(x_val)
   
   # Cross-validation:
    scoring = {'accuracy': make_scorer(accuracy_score), "f1": make_scorer(f1_score)}
    cross_val = cross_validate(model, x_train, y_train, cv=5, scoring=scoring)
    cross_vals.append([np.mean(x) for x in list(cross_val.values())])
    accuracy_scores = cross_val['test_accuracy']
    f1_scores = cross_val['test_f1']
    
    #Metrics:
    f1 = f1_score(y_val, prediction, average='macro').round(3)
    prec = precision_score(y_val, prediction, average='macro').round(3)
    rec = recall_score(y_val, prediction, average='macro').round(3)
    acc_sq = accuracy_score(y_val, prediction).round(3)
    report = classification_report(y_val, prediction, target_names=classes)
    loss = log_loss(y_val, prediction_prob).round(3)
    roc_auc = roc_auc_score(y_val, prediction).round(3)
    pr_auc = average_precision_score(y_val, prediction).round(3)
    
   #Prints:
    print (estimator.__class__.__name__)
    conf = confusion_matrix(model, x_train, y_train, x_val, y_val, classes=classes,
        cmap="YlGn", size=size, **kwargs)
    print("Classification Report:")
    print(report)
    print (f'Cross-validation')
    print ("Accuracy scores: {}".format(accuracy_scores))
    avg_list = sum(list(accuracy_scores))/len(list(accuracy_scores))
    print (f"Accuracy score (average): {avg_list}")
    print ("F1 scores for 'Yes' values: {}".format(f1_scores))
    print (f"Average F1 score: {f1}")
    print (f"ROC-AUC score: {roc_auc}")
    print (f"PR-AUC score: {pr_auc}")
    print (f"Log-loss: {loss}")
    print('')
    
    #Appends:
    models_list.append(model)
    predictions[f'{estimator.__class__.__name__}'].append(prediction)
    predictions_prob[f'{estimator.__class__.__name__}'].append(prediction_prob)
    df["model_name"].append(f'{estimator.__class__.__name__}')
    df["a_score"].append(acc_sq)
    df["f1_score"].append(f1)
    df["precision_score"].append(prec)
    df["recall_score"].append(rec)
    df["ROC_AUC_score"].append(roc_auc)
    df["PR_AUC_score"].append(pr_auc)
    df["loss"].append(loss)
    return model
    


def get_empty_lists_and_dfs():
    cross_vals = []
    preprocessors = []
    models = []
    predictions = {}
    predictions_prob = {}
    metrics_df = {}
    metrics_df["precision_score"] = []
    metrics_df["recall_score"] = []
    metrics_df["model_name"] = []
    metrics_df["a_score"] = []
    metrics_df["f1_score"] = []
    metrics_df["ROC_AUC_score"] = []
    metrics_df["PR_AUC_score"] = []
    metrics_df["loss"] = []
    metrics_df["exec_time"] = []
    metrics_df["encoders"] = []
    metrics_df["cimputers"] = []
    metrics_df["nimputers"] = []
    metrics_df["scalers"] = []
    metrics_df["num_features"] = []
    metrics_df["cat_features"] = []
    metrics_df["bin_features"] = []
    metrics_df["other_features"] = []
    optuna_df = {}
    optuna_df["best_ROC_AUC_score"] = []
    optuna_df["best_params"] = []
    optuna_df["exec_time"] = []
    return cross_vals, models, predictions, predictions_prob, metrics_df, preprocessors, optuna_df


def run_models(metrics_df, x_train, y_train, x_val, y_val, features,
               numeric_features_list,
               binary_features_list,
               categorical_features_list,
               other_features_list,
              scalers_list,
              num_imputers_list,
               cat_imputers_list,
              encoders_list, 
              classifiers):
                                                                                                                   
    # Running pipelines on various combinations of transformers and features:

    for numeric_features, binary_features, categorical_features, other_features in zip(numeric_features_list,
                                                                                       binary_features_list,
                                                                                       categorical_features_list,
                                                                                       other_features_list):
        for scaler in scalers_list:
            for nimputer, cimputer in zip(num_imputers_list, cat_imputers_list):
                for encoder in encoders_list:

                    num_transformer, bin_transformer, cat_transformer, imp_transformer = get_transformers(scaler, 
                                                                                                          nimputer,
                                                                                                          cimputer,
                                                                                                          encoder)
                    preprocessor = ColumnTransformer(
                           transformers=[
                            ('numeric', num_transformer, numeric_features.content),
                            ('binary', bin_transformer, binary_features.content),
                            ('categorical', cat_transformer, categorical_features.content),
                            ('other', imp_transformer, other_features.content)
                        ])
                    preprocessors.append(preprocessor)

                    print(f'Parameters for the dataset and transformers: {encoder.__class__.__name__}, \
{nimputer}, {cimputer}, {scaler.__class__.__name__}, {numeric_features.name}, \
{binary_features.name}, {categorical_features.name}, {other_features.name}')

                    for classifier in classifiers:

                        predictions[f'{classifier.__class__.__name__}'] = []
                        predictions_prob[f'{classifier.__class__.__name__}'] = []
                        start_time1 = time.time()
                        model = score_model(X_train, y_train, X_val, y_val, classifier, preprocessor, metrics_df, 
                                    models, classes_target, size_target, features)
                        end_time1 = time.time()
                        exec_time1 = end_time1 - start_time1
                        metrics_df["exec_time"].append(exec_time1)
                        metrics_df["encoders"].append(encoder.__class__.__name__)
                        metrics_df["nimputers"].append(nimputer.__class__.__name__)
                        metrics_df["cimputers"].append(cimputer.__class__.__name__)
                        metrics_df["scalers"].append(scaler.__class__.__name__)
                        metrics_df["num_features"].append(numeric_features.name)
                        metrics_df["cat_features"].append(categorical_features.name)
                        metrics_df["bin_features"].append(binary_features.name)
                        metrics_df["other_features"].append(other_features.name)
                        print('')
                        print(f'Execution time: {exec_time1}')
                        joblib.dump(model, f'model_{classifier.__class__.__name__}_\
{encoder.__class__.__name__}_{nimputer}_{cimputer}_\
{numeric_features.name}.joblib')
                        print('')
    metrics_df = pd.DataFrame(metrics_df)
    metrics_df.to_csv(f'homecredit_features_{features}.csv')
    

def set_objective(trial, X_train, y_train, X_val, y_val):
    classifier_name = trial.suggest_categorical('classifier', ['XGB', 'RF', 'ET', 'GB', 'LR', 'KNN', 
                                                               'Bagging', 'AdaBoost'])

    if classifier_name == 'XGB':
        classifier = xg.XGBClassifier(
        n_estimators = trial.suggest_int("n_estimators", 100,1000, step = 100),
        max_depth = trial.suggest_int('max_depth', 3, 9, step = 3),
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log = True),
        subsample = trial.suggest_float('subsample', 0.6, 1, step = 0.2)
        )
        classifier.fit(X_train, y_train)

    elif classifier_name == 'RF':
        classifier = RandomForestClassifier(
       # max_features = trial.suggest_int("max_features", 6,32, step = 5),
        n_estimators = trial.suggest_int("n_estimators", 10,101, step = 10),
        max_depth = trial.suggest_int("rf_max_depth", 2, 64, log=True),
        max_samples = trial.suggest_float("max_samples",0.2, 1),
        random_state = 42
        )
        classifier.fit(X_train, y_train)
        
    elif classifier_name == 'ET':
        classifier = ExtraTreesClassifier(
        n_estimators = trial.suggest_int("n_estimators", 100,500, step = 200),
    #    max_features = trial.suggest_int("max_features", 6,32, step = 5),
        max_depth = trial.suggest_int('max_depth',1, 9, step = 4),
        )
        classifier.fit(X_train, y_train)
        
    elif classifier_name == 'GB':
        classifier = GradientBoostingClassifier(
        n_estimators = trial.suggest_int("n_estimators", 100,500, step = 200),
        max_depth = trial.suggest_int("rf_max_depth", 2, 64, log=True),
     #   max_features = trial.suggest_int("max_features", 6,32, step = 5),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log = True)
        )
        classifier.fit(X_train, y_train)
        
    elif classifier_name == 'LR':
        classifier = LogisticRegression(
            C=trial.suggest_float('C', 1, 9, step = 4),
            penalty=trial.suggest_categorical('penalty', ['l1', 'l2']),
            solver=trial.suggest_categorical('solver', ['liblinear', 'saga'])
        )
        classifier.fit(X_train, y_train)
        
    elif classifier_name == 'KNN':
        classifier = KNeighborsClassifier(
            n_neighbors=trial.suggest_int('n_neighbors', 3, 7, step = 2),
            weights=trial.suggest_categorical('weights', ['uniform', 'distance']),
            leaf_size=trial.suggest_int('leaf_size', 30, 90, step = 30)
        )
        classifier.fit(X_train, y_train)
        
    elif classifier_name == 'Bagging':
        classifier = BaggingClassifier(
            n_estimators=trial.suggest_int('n_estimators', 10, 90, step = 20),
       #     max_features = trial.suggest_int("max_features", 6,32, step = 5),
            max_samples=trial.suggest_float('max_samples', 0.5, 0.9, step = 0.2),
        )
        classifier.fit(X_train, y_train)
        
    else:
        classifier = AdaBoostClassifier(
            n_estimators=trial.suggest_int('n_estimators', 50, 200, step = 50),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 10.0, log = True)
        )
        classifier.fit(X_train, y_train)
    
    y_pred = classifier.predict(X_val)
    score = roc_auc_score(y_val, y_pred)

    return score



def run_optuna():
    cross_vals, models, predictions, predictions_prob, scores, preprocessors, optuna_df = get_empty_lists_and_dfs()

    # Running pipelines on various combinations of transformers and features:

    for numeric_features, binary_features, categorical_features, other_features in zip(numeric_features_list,
                                                                                       binary_features_list,
                                                                                       categorical_features_list,
                                                                                       other_features_list):
        for scaler in scalers_list:
            for nimputer, cimputer in zip(num_imputers_list, cat_imputers_list):
                for encoder in encoders_list:

                    num_transformer, bin_transformer, cat_transformer, imp_transformer = get_transformers(scaler, 
                                                                                                          nimputer,
                                                                                                          cimputer,
                                                                                                          encoder)
                    preprocessor = ColumnTransformer(
                           transformers=[
                            ('numeric', num_transformer, numeric_features.content),
                            ('binary', bin_transformer, binary_features.content),
                            ('categorical', cat_transformer, categorical_features.content),
                            ('other', imp_transformer, other_features.content)
                        ])
                    preprocessors.append(preprocessor)

                    # Hyperparameter tuning - Bayesian optimization:

                    if isinstance(encoder, WOEEncoder):
                        X_pipe_train = pd.DataFrame(preprocessor.fit_transform(X_train, y_train))
                        X_pipe_val = pd.DataFrame(preprocessor.fit_transform(X_val, y_val))

                        randomundersampler = RandomUnderSampler(random_state=42)
                        X_resampled_train, y_resampled_train = randomundersampler.fit_resample(X_pipe_train, y_train)

                        def objective(trial):
                            return set_objective(trial, X_resampled_train, y_resampled_train, X_pipe_val, y_val)

                        start_time2 = time.time()
                        optuna.logging.set_verbosity(optuna.logging.WARNING)
                        study = optuna.create_study(direction = "maximize")
                        study.optimize(objective, n_trials = 100)
                        trial = study.best_trial
                        end_time2 = time.time()
                        exec_time2 = end_time2 - start_time2
                        print("Best ROC-AUC score by the Bayesian optimization (Optuna): ", trial.value)
                        print("Best parameters of classifiers: ")
                        for key, value in trial.params.items():
                            print("  {}: {}".format(key, value))
                        optuna_df['best_ROC_AUC_score'].append(trial.value)
                        optuna_df['best_params'].append(trial.params)
                        optuna_df['exec_time'].append(exec_time2)
                        print(f'Execution time for the Bayesian optimization (Optuna): {exec_time2}')
                        print('')



def find_inputs(model, x_train, y_train, x_val, y_val, list_length): 
    data = {}
    var_list = [0 + i * 1 for i in range(list_length)]
    i_list = [[]]
    u = 0
    while u < 5000:
        i = random.choices(var_list[:-1], k=random.choice(var_list[1:]))
        i = list(np.unique(i))
        x = x_train.iloc[:,i]
        mod = model.fit(x, y_train)
        prediction = mod.predict(x_val.iloc[:,i])
        roc_auc = roc_auc_score(y_val, prediction)
        if i not in i_list:
            data[f"{i}"] = roc_auc.round(3)
            i_list.append(i)
        u+=1
    
    print('Combinations of independent variables for the model with the highest roc-auc scores:')
    print(f'{max(data, key=data.get)}')
    print('Roc-auc score:') 
    print(data[f'{max(data, key=data.get)}'])
    return data


def test_model(estimator, x_test, y_test, classes):
    prediction = estimator.predict(x_test)
    prediction_prob = estimator.predict_proba(x_test)

    #Metrics:
    f1 = f1_score(y_test, prediction, average='macro').round(3)
    prec = precision_score(y_test, prediction, average='macro').round(3)
    rec = recall_score(y_test, prediction, average='macro').round(3)
    acc_sq = accuracy_score(y_test, prediction).round(3)
    report = classification_report(y_test, prediction, target_names=classes)
    loss = log_loss(y_test, prediction_prob).round(3)
    roc_auc = roc_auc_score(y_test, prediction).round(3)
    pr_auc = average_precision_score(y_test, prediction).round(3)
    conf_matrix = cfm(y_test, prediction)
    
   #Prints:
    print (estimator[3].__class__.__name__)
    plot_confusion_matrix(conf_matrix, classes, (3,2))
    print("Classification Report:")
    print(report)
 
    print (f"Accuracy score: {acc_sq}")
    print (f"F1 score: {f1}")
    print (f"ROC-AUC score: {roc_auc}")
    print (f"PR-AUC score: {pr_auc}")
    print (f"Log-loss: {loss}")


def fit_pca(data, list_of_features, number_of_components, pc_column_names):
    
    # fit data:
    pca = PCA(n_components=number_of_components)
    pca.fit(data.loc[:, list_of_features].fillna(0))
    
    # transform data:
    data_transformed = pd.DataFrame(pca.transform((data.loc[:, list_of_features].fillna(0))), 
                columns=[f'PC{i+1}' for i in range(number_of_components)], index=data.index)
    
    # print outputs:
    print(f'Explained variance ratio: {pca.explained_variance_ratio_}')
    print('')
    print(f'Singular values: {pca.singular_values_}')
    print('')
    print(f'Components: {pca.components_}')
    
    # get principal components table:
    pca_df = pd.DataFrame(pca.components_)
    pca_df.columns = list_of_features
    pca_df = pca_df.transpose().round(3)
    pca_df.columns = pc_column_names
    pca_df = pca_df.sort_values(pc_column_names[0],ascending = False)
    
    # plot explained variance ratios:
    cum_sum_eigenvalues = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(6, 3))
    plt.bar(range(0,len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_, 
            alpha=0.5, align='center', label='Individual explained variance')
    plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',
             label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show();
    
    return pca_df, data_transformed


def binary_numeric_or_zeros(data):
    unique_values = data.unique()
    return len(unique_values) <= 2


def get_binary_numeric_or_zeros(data):
    data_describe = data.describe()
    data_num =  data.loc[:,data_describe.columns]
    data_bin_num = data.loc[:, data.apply(binary_numeric_or_zeros)]
    if len(data_bin_num.columns) != 0:
        return data_bin_num
    else:
        pass


def value_counts_categorical(data):
    data_describe = data.describe()
    data_cat = data.copy()
    for i in range(len(data_describe.columns)):
        data_cat =  data_cat.drop(columns = [data_describe.columns[i]])   
    for i in range(len(data_cat.iloc[0,:])):
        print(data_cat.iloc[:,i].value_counts())
        print('')


def value_counts_binary_numeric(data):
    data_describe = data.describe()
    data_bin_num = get_binary_numeric(data)
    if type(data_bin_num) is None.__class__:
        print('There are no nonary numeric variables in the dataset.')
    else:
        for i in range(len(data_bin_num.iloc[0,:])):
            print(data_bin_num.iloc[:,i].value_counts())
            print('')


def remove_duplicates(main_list, list1, list2, list3):
    combined_lists = list1 + list2 + list3
    result_list = [item for item in main_list if item not in combined_lists]
    return result_list


def get_correlations(data):
    correlation_matrix = data.iloc[:, :-1].corr()

    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    sns.heatmap(correlation_matrix, annot=True, cmap='RdYlGn', 
                vmin=-1, vmax=1, linewidths=0.2, mask=mask)

    fig = plt.gcf()
    fig.set_size_inches(20, 20)

    plt.show();


def plot_bars(data,y,y_label,title, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x=data.index, y=y,
            data=data, 
            errorbar="sd", color = sns.color_palette()[1]).set(title=title)
    plt.xticks(rotation=90)
    ax.bar_label(ax.containers[0])
    ax.set_ylabel(y_label)
    ax.set_xlabel("")
    plt.show()


def check_missing_values(file):
    check = numpy.where(file.isnull())
    check = pd.DataFrame(check)
    for i in range(len(check.iloc[0,:])):
        print(f'missing value in the row {check.iloc[0,i]} of the column {check.iloc[1,i]}.')
    print(f'The total number of missing values is: {len(check.axes[1])}')

def find_outliers(dataframe,x,coef):
    '''
    The function to find outliers in numerical variables on the basis of IQR rule
    '''
    count_lower = []    
    count_upper = []
    Q1 = dataframe.iloc[:,x].quantile(0.25)
    Q3 = dataframe.iloc[:,x].quantile(0.75)
    IQR = Q3 - Q1
    lower_lim = Q1 - coef*IQR
    upper_lim = Q3 + coef*IQR
    for data in range(len(dataframe.iloc[:,0])):
        if dataframe.iloc[data,x] < lower_lim:
            count_lower.append(data)
        elif dataframe.iloc[data,x] > upper_lim:
            count_upper.append(data)
    print(f'The variable: {dataframe.columns[x]}')
    print(f'The number of lower outliers is:{len(count_lower)},\
    The number of upper outliers is :{len(count_upper)}')
    print('')

def stacked_bars(file, title_label, title):
    ax = file.plot(kind="barh", stacked=True, rot=0)
    ax.legend(title=title_label, bbox_to_anchor=(1, 1.02),
             loc='upper left')
    plt.xlabel("")
    plt.xticks(rotation = "vertical")
    plt.xlabel("%")
    for c in ax.containers:
        ax.bar_label(c, label_type='center')
    plt.title(title)
    plt.show();

def plot_confusion_matrix(conf_matrix, classes, figsize):
    plt.figure(figsize=figsize)
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGn",
                xticklabels=classes_target, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show();


def chi_square_test(data, confidence, variable):
    stat, p, dof, expected = stats.chi2_contingency(data)
    alpha = 1 - confidence
    print(f'Pearson chi square test:{stat.round(3)}')
    print(f'P_value: {p.round(3)}')
    return p


def get_transformers (scaler, nimputer, cimputer, encoder):
    num_transformer = Pipeline([
          ('scaler', scaler),
          ('imputer', nimputer)])
    
    bin_transformer = Pipeline(steps=[
          ('ordinal', WOEEncoder()),
          ('imputer', nimputer)])

    cat_transformer = Pipeline(steps=[
          ('encoder', encoder),
          ('imputer', cimputer)])

    imp_transformer = Pipeline(steps=[
       ('imputer', nimputer)])
    
    return num_transformer, bin_transformer, cat_transformer, imp_transformer


