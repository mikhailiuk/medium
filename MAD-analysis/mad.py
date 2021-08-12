import numpy as np
import pandas as pd

def select_pairs(df_in, gt_name):
        
    n = len(np.arange(0,len(df_in)))
        
    methods = list(df_in.columns)
    data_df =  pd.DataFrame(columns=['idx_1', 'idx_2']+methods, data=np.zeros((int(n/2*((n-1))),len(methods)+2))) 
    
    count= 0

    # oi - oj
    for ii in range(0,len(df_in)):
        for jj in range(ii+1,len(df_in)):
            
            data_df['idx_1'][count] = ii
            data_df['idx_2'][count] = jj
            for method in methods:
                data_df[method][count] =  df_in[method][ii]-df_in[method][jj]
            count+=1 
            
    # difference between the metrics
    for method_1 in methods:
        for method_2 in methods:
            if method_1!=method_2 and (not (gt_name in [method_1,method_2])):
                data_df[method_1+'_'+method_2] = abs(data_df[method_1])-abs(data_df[method_2])
    
    return data_df



def run_mad(data_df, methods, gt_name, numb_ex = 50, threshold_less = 0.1):
    
    result_array = np.zeros((len(methods)-1,len(methods)-1))
    methods_no_gt = [meth for meth in methods if meth!=gt_name]
    
    # Go over all methods
    for tt, method_1 in enumerate(methods_no_gt):
        for kk, method_2 in enumerate(methods_no_gt):
            
            list_of_ids = []
            wins = 0
            
            # Do not consider comparisons to itself and the ground truth
            if method_1 != method_2 and (not (gt_name in [method_1,method_2])):
                
                
                # Sort in descending order i.e. far according to method 1 and close according method 2
                data_df.sort_values(by=[method_1+'_'+method_2],ascending=False,inplace=True)
                data_df.reset_index(inplace= True, drop=True)
                
                # Find conditions that are close according to method 2
                data_df_slice = data_df.loc[abs(data_df[method_2])<threshold_less]
                data_df_slice.reset_index(inplace= True, drop=True)
                
                
                # go over the slice
                for ii in range(0,len(data_df_slice)):
                    
                    # Consider only the first numb_ex points
                    if (len(list_of_ids)/2)>numb_ex:
                        break
                        
                    
                    # Make sure that each condition is used only once
                    if (data_df_slice['idx_1'][ii] not in list_of_ids) and (data_df_slice['idx_2'][ii] not in list_of_ids):

                        list_of_ids.extend([data_df_slice['idx_1'][ii],data_df_slice['idx_2'][ii]])
                        
                        
                        # If the difference is greater than the considered threshold according to method 1 and gt
                        # and method 1 correctly ranks conditions add one to the number of won points
                        if (abs(data_df_slice[gt_name][ii])>threshold_less and data_df_slice[gt_name][ii]*data_df_slice[method_1][ii]>0):
                            wins+=1

                result_array[tt][kk] = wins/(len(list_of_ids)/2)
                
    return result_array



# Order the metrics in the matrix according to their score
def order_scores(result_array, methods):
    sumarr  = np.sum(result_array,1)
    correct_order = np.flip(sumarr.argsort())

    metric_names_corr_order = []
    for ii in correct_order:
        metric_names_corr_order.append(methods[ii])

    result_array_sorted = np.zeros((len(methods),len(methods)))

    for ii in range(0,len(sumarr)):
        for jj in range(0,len(sumarr)):
            result_array_sorted[ii,jj] = result_array[correct_order[ii],correct_order[jj]]
            
    return result_array_sorted, metric_names_corr_order
