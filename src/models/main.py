import sys
import os

project_root = os.path.abspath("/home/hajali/Desktop/Bargh_Ml_project/")
sys.path.insert(0, project_root)

from true_generation_model import Model
from status_transformer import StatusTransformer
from best_model import BestModel
from feature_selector import FeatureSelector
import pandas as pd
import yaml
from lstm import StatusLSTM
import time
import matplotlib.pyplot as plt
from clustering import Cluster
from classifier import Classifer
from pipeline import Pipeline
from seasonality import Seasonality

from wakepy import keep
import time
import gc
import psutil


def load_file(filename):
    with open(filename, 'r') as f:
        return yaml.safe_load(f)


if __name__ == "__main__":

    start_time = time.perf_counter()

    df_original = pd.read_csv(filepath_or_buffer='/home/hajali/Desktop/Bargh_Ml_project/data/processed/operation.csv')
    df = pd.get_dummies(df_original, columns=['value'], drop_first=True)
    # df = df[((df['status'] == 'SO') | (df['status'] == 'LF1')) & (df['code'].str.contains("G"))]
    df = df[((df['status'] == 'SO') | (df['status'] == 'LF1'))]

    s = Seasonality()
    df = s.add_seasonality_effect(df=df)

    bst = BestModel()

    powerplants = df[['name', 'code']].drop_duplicates()
    # powerplants = powerplants[powerplants['name'] == "حافظ"]
    # powerplants = df[['name']].drop_duplicates()

    config = load_file('/home/hajali/Desktop/Bargh_Ml_project/configs/status_transformer.yaml')

    selected_features = ['temperature', 'value_P', 'value_M', 'surface_pressure', 'humidity', 'forecast', 'commitment', 'year_effect', 'week_effect', 'hour_effect']

    drp = [item for item in config['feature_cols'] if item not in selected_features]

    # bst.update_classifiers()
    
    model = Model(df=df, target='generation', features_to_drop=drp)

    # cluster = Cluster()
    # df_labeled = cluster.gaussian_cluster(data=df, n=3)
    # df['group'] = 0

    
    # for _, item in powerplants.iterrows():
        # cls.logistic(name=item['name'], code=item['code'])
        # cls.RFC(name=item['name'], code=item['code'])
        # cls.LDA(name=item['name'], code=item['code'])
        # cls.QDA(name=item['name'], code=item['code'])
        # cls.KNN(name=item['name'], code=item['code'])
        # cls.SMC(name=item['name'], code=item['code'])
        # cls.MLP(name=item['name'], code=item['code'])
        # cls.XGB(name=item['name'], code=item['code'])
        # break
    
    # predefined = os.listdir("/home/hajali/Desktop/Bargh_Ml_project/models/classifiers")

    # with keep.running():
    #     try:
    #         counter = 0
    #         for _, item in powerplants.iterrows():

    #             cls = Classifer(df=df, target='group', features_to_drop=drp, base_path="/home/hajali/Desktop/Bargh_Ml_project/models/classifiers")

    #             continue_check = False
    #             for dirname in predefined:
    #                 if f"{item['name']}-{item['code']}" in dirname:
    #                     continue_check = True
    #             if continue_check:
    #                 print(f"Pass the unit: {item['name']}-{item['code']}")
    #                 continue

    #             for method in ['generation', 'features']:
    #                 for cluster in ['gaussian', 'kmeans']:
    #                     for n in [2, 3, 4]:

    #                         cls.update_class(
    #                             method=method,
    #                             cluster=cluster,
    #                             n=n
    #                         )

    #                         cls.logistic(name=item['name'], code=item['code'])
    #                         cls.KNN(name=item['name'], code=item['code'])
    #                         cls.LDA(name=item['name'], code=item['code'])
    #                         cls.MLP(name=item['name'], code=item['code'])
    #                         cls.QDA(name=item['name'], code=item['code'])
    #                         cls.RFC(name=item['name'], code=item['code'])
    #                         cls.SMC(name=item['name'], code=item['code'])
    #                         cls.XGB(name=item['name'], code=item['code'])

    #                         print("collecting garbages.")
    #                         gc.collect()
                
    #             bst.choose_best_classifier(name=item['name'], code=item['code'])

    #             del cls

    #             gc.collect()

    #             counter += 1

    #             if counter == 3:
    #                 with open("condition.txt", "w") as f:
    #                     f.write("again")
    #                 break

    #     except Exception as e:
    #         print({e})

    #     if counter == 0:
    #         with open("condition.txt", "w") as f:
    #                     f.write("stop")
        
    #     end_time = time.perf_counter()
    #     elapsed = end_time - start_time
    #     print("-----------------------------------\n")
    #     print(f"Elapsed time: {elapsed:.4f} seconds\n")
    #     print("-----------------------------------")
    
    # now = time.time()

    # for _, item in powerplants.iterrows():
    #     model.kernel_ridge(name=item['name'], code=item['code'], kernel='rbf', gamma=0.6, alpha=0.2, save_model=True)
    #     time.sleep(5)
    
    # print(f"Time consumes: {time.time() - now}")
    # model.kernel_ridge(name='عسلویه', code=None, kernel='rbf', gamma=0.6, alpha=0.2, save_model=True)

    # for _, item in powerplants.iterrows():
    #     print(f"{item['name']}  -  {item['code']}")

    for _, item in powerplants.iterrows():
        model.random_forest(name=item['name'], code=item['code'], n_estimator=150, depth=80, model_save_path=True)
        model.xgboost(name=item['name'], code=item['code'], n_estimator=200, depth=80, epochs=2000, learning_rate=0.04, save_model=True)
        model.linear(name=item['name'], code=item['code'], gd=False, save_model=True)
        model.polynomial(name=item['name'], code=item['code'], degree=3, gd=False, save_model=True)
        print("-------------------------------------------------------")
        print("-------------------------------------------------------")
        time.sleep(5)
    
    
    # for _, item in powerplants.iterrows():
    #     for feature in config['feature_cols']:
    #         model = Model(df=df, target='generation', features_to_drop=[feature])
    #         model.random_forest(name=item['name'], code=None, n_estimator=150, depth=80, model_save_path=True)
    #         model.xgboost(name=item['name'], code=None, n_estimator=200, depth=80, epochs=2000, learning_rate=0.04, save_model=True)
    #         model.linear(name=item['name'], code=None, gd=False, save_model=True)
    #         model.polynomial(name=item['name'], code=None, degree=3, gd=False, save_model=True)
    #         print("-------------------------------------------------------")
    #         print("-------------------------------------------------------")
    #         time.sleep(5)

    # model.linear(
    #     name='حافظ', code='G14', penalty='l2', epochs=1000, initial_lr=0.0001, lr="invscaling",
    #     max_iter=1, save_model=True
    # )

    # model.linear(name='حافظ', code=None, gd=False, save_model=True)

    # model.polynomial(name='حافظ', code='G14', degree=3, penalty=None, epochs=2000, initial_lr=0.00002,
    #                  lr='constant', max_iter=3, save_model=True)

    # model.polynomial(name="حافظ", code=None, gd=False, degree=2, save_model=True)

    # model.random_forest(name='حافظ', code=None, n_estimator=150, depth=80, model_save_path=True)

    # model.xgboost(name='حافظ', code=None, n_estimator=200, depth=80, epochs=2000, learning_rate=0.04, save_model=True)

    # model.polynomial(name='شهدای پیروز - بهبهان', code='G11', model_save_path=True, fig_save_path=True, penalty='l2')
    # model.random_forest(name='شهدای پیروز - بهبهان', code='G11', model_save_path=True)
    # model.xgboost(name='شهدای پیروز - بهبهان', code='G11', model_save_path=True, fig_save_path=True, epochs=10000, learning_rate=0.1)

    # transformer = StatusTransformer(dataframe=df_original, feature_cols=config['feature_cols'], target_col=config['target'])
    # transformer.pipeline(name='شهدای پیروز - بهبهان', code='G11', lr=0.01, epochs=2, save=True)

    # model = StatusLSTM(dataframe=df_original, feature_cols=config["feature_cols"], target_col=config["target"])

    # model.pipeline(name='شهدای پیروز - بهبهان', code='G11', lr=1e-3, weight_decay=0.0, epochs=100, save=True)

    # y = model.predict(df.loc[[167]], "/home/hajali/Desktop/Bargh_Ml_project/models/حافظ-G14-linear#8")
    # print(df.loc[[167]]['generation'])
    # print(f"Predicted y with linear is: {y}")

    # y = model.predict(df.loc[[167]], "/home/hajali/Desktop/Bargh_Ml_project/models/حافظ-G14-randomforest#6")
    # print(f"Predicted y with ranomforest is: {y}")

    # df_test = df[(df['name']=='حافظ') & (df['code']=='G14')]
    # for i, (idx, row) in enumerate(df_test.iterrows()):
    #     row = row.to_frame().T
    #     y = model.predict(row, "/home/hajali/Desktop/Bargh_Ml_project/models/حافظ-G14-randomforest#6")
    #     print(f"Input is: {row['generation']}")
    #     print(f"Prediction with randomforest is: {y}")
    #     print("--------------------------------")
    #     if (i+1) % 50 == 0:
    #         break
    # best = BestModel()
    # best.update_classifiers()

    # for _, item in powerplants.iterrows():
    #     df_test = df[(df['name']==item['name']) & (df['code']==item['code'])]
        
    #     y_pred = best.best_prediction(df_test, byName=True)

    #     print(model.evaluate_unit(df_test, predictions=y_pred))

    #     model.plot_prediction(name=item['name'], code=item['code'], predictions=y_pred)

    # for i in range(0, len(df_test.index), 48):
    #     plt.plot(df_test['date'][i:i+48], df_test['generation'][i:i+48], label='Actual')
    #     plt.plot(df_test['date'][i:i+48], y_pred[i:i+48], label='Prediction')
    #     plt.legend()
    #     plt.xlabel('Time')
    #     plt.ylabel('Generation')
    #     plt.xticks(rotation=90)
    #     plt.show()
    # feature_counter = {}
    # for _, item in powerplants.iterrows():
    #     df_copy = df[df['name']==item['name']]
    #     df_copy = df_copy.drop(columns=['id', 'hour', 'date', 'name', 'code', 'status', 'declare'], axis=1)
        
    #     fs = FeatureSelector(df_copy.drop(columns=['generation']), df_copy['generation'])
    #     print(f"Scores for powerplant {item['name']}")
    #     anova = list(fs.ANOVA_Fvalue(5))
    #     for f in anova:
    #         if f in feature_counter.keys():
    #             feature_counter[f] += 1
    #         else:
    #             feature_counter[f] = 1
    #     print(f"ANOVA F-value: {anova}")

    #     lasso = list(fs.lasso_selector(alpha=0.01, threshold=1e-6))
    #     for f in lasso:
    #         if f in feature_counter.keys():
    #             feature_counter[f] += 1
    #         else:
    #             feature_counter[f] = 1
    #     print(f"Lasso selector: {lasso}")

    #     mi = list(fs.mutual_info_reg(5))
    #     for f in mi:
    #         if f in feature_counter.keys():
    #             feature_counter[f] += 1
    #         else:
    #             feature_counter[f] = 1
    #     print(f"Mutual information: {mi}")

    #     rfe = list(fs.RFE(5))
    #     for f in rfe:
    #         if f in feature_counter.keys():
    #             feature_counter[f] += 1
    #         else:
    #             feature_counter[f] = 1
    #     print(f"RFE: {rfe}")

    #     tree = list(fs.tree_selector(5))
    #     for f in tree:
    #         if f in feature_counter.keys():
    #             feature_counter[f] += 1
    #         else:
    #             feature_counter[f] = 1
    #     print(f"Tree selector: {tree}")
    #     print()
    #     print()
    
    # print(feature_counter)

    # cluster = Cluster(features=drp, base_path="/home/hajali/Desktop/Bargh_Ml_project/src/visualization/unit_figs/Clustering/Features-all")
    # for i in [3, 4]:
    #     for _, item in powerplants.iterrows():

            # df_test = df[(df['name']==item['name']) & (df['code']==item['code'])]
            # df_test = df_test.dropna()

            # cluster.mean_cluster(data=df_test)

            # cluster.kmeans_cluster(data=df_test, n=i, save=False)

            # cluster.gaussian_cluster(data=df_test, save=False, n=i)
            # break
    #         # print(df_labeled.head())

    #         # cluster.hmm(sample=df_test)
            # break
    
    # p = Pipeline(df=df, features_to_drop=drp)
    # for _, item in powerplants.iterrows():
    #     p.by_bar(name=item['name'], code=item['code'], save=True)
    # best = BestModel()
    # best.update()

    # methods = ['generation', 'features']
    # clusters = ['gaussian', 'kmeans']

    # for _, item in powerplants.iterrows():
    #     p.execute(name=item['name'], code=item['code'], save=True)
        
        # df_test = df[(df['name']==item['name']) & (df['code']==item['code'])]
        # best = BestModel()
        # y_pred = best.best_prediction(df_test)

        # print(model.evaluate_unit(df_test, predictions=y_pred))

        # model.plot_prediction(name=item['name'], code=item['code'], predictions=y_pred, save=False)


    # df_test = df[(df['name']=='حافظ') & (df['code']=='G14')]
    # df_test.hist(column='generation', bins=15)

