import pandas as pd
import numpy as np

def results_analysis(args,setting_analysis_name,test_dataset,flag):
    all_df_metrics_rul = pd.DataFrame()

    for k in range(args.itr):  #args.itr
        file_name = setting_analysis_name + '{}'.format(k)

        metrics_rul = np.load(args.root_path + 'results/{}_{}/{}/{}_{}_rul.npy'.format(args.train_dataset_name,args.test_dataset_name,file_name,test_dataset,flag))

        df_metrics_rul = pd.DataFrame(data = [metrics_rul], columns = ['score', 'rmse', 'prec', 'mse', 'mae', 'r2' ]) 
        all_df_metrics_rul = all_df_metrics_rul.append(df_metrics_rul)

    means_rul = pd.DataFrame([all_df_metrics_rul.mean()],index=['mean'])
    all_df_metrics_rul = all_df_metrics_rul.append(means_rul)
    all_df_metrics_rul['RUL'] = ['rul']*(len(all_df_metrics_rul)-1) + ['mean']
    all_df_metrics_rul.drop(['RUL','mse', 'mae', 'r2'],axis=1,inplace=True)

    print('-------------mean_rul_{}----------------'.format(flag))
    print(all_df_metrics_rul.iloc[-1,:])

    return  all_df_metrics_rul
