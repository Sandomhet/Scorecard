import pandas as pd
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

#=========Reading data=======================

raw_data = pd.read_excel(r'C:\\Users\\apk15\\Desktop\t_score_card_grp.xlsx')
is_bad_col_name  =  'is_bad' 
 
#====Change the data to WOE==================

'''Get the table of woe'''
def get_woe_tb(data,is_bad_col_name):
    woe_tb = pd.DataFrame(columns=['feature','grp','woe'])
    
    sample_num = data.shape[0]
    bad_tt = data[is_bad_col_name].sum()
    godd_tt = sample_num- bad_tt
        
    for col, row in data.items():
        if(col !=is_bad_col_name):   
            woe_df =data.groupby(col)[is_bad_col_name].agg([('bad_cn','sum'),('cn','count')])
            woe_df['grp'] = woe_df.index
            woe_df.loc[woe_df['bad_cn']==0,'bad_cn']=1
            woe_df.loc[woe_df['bad_cn']==sample_num,'bad_cn'] -=1
            woe_df['woe']=np.log((woe_df['bad_cn']/bad_tt)/((woe_df['cn'] -woe_df['bad_cn'])/godd_tt))
            woe_df['feature'] = col
            woe_tb=woe_tb._append(woe_df[['feature','grp','woe']],ignore_index=True)
    return woe_tb


'''change the data to woe'''     
def data_to_woe(data,is_bad_col_name):
    woe_data = data.copy()
    woe_tb = get_woe_tb(data,is_bad_col_name)
    for col, row in woe_data.items():
        if(col !=is_bad_col_name):   
            woe_df =woe_tb[woe_tb['feature']==col]
            woe_dict = {woe_df['grp'][col]:woe_df['woe'][col] for col in woe_df.index}
            woe_data[col] = woe_data[col].map(woe_dict)
    return woe_data,woe_tb

# ---change data to woe---
data,woe_tb_all = data_to_woe(raw_data,is_bad_col_name)  # data here is the data of woe，woe_tb is the woe of each eigenvalue
y    = np.array(data[is_bad_col_name])
X    = np.array(data.drop(is_bad_col_name, axis=1))
feature_names = data.columns[data.columns!=is_bad_col_name]

#==============Model Training===========================
'''Choose the varible one by one'''
def select_feture(clf,X,y,feature_names):
    select_fetures_idx = []                            # The current varible we choose  
    var_pool   = np.arange(X.shape[1])                           # The rest varible
    auc_rec    = []
    print("\n===========The course of recurssion===============")
    while(len(var_pool)>0):
        max_auc  = 0
        best_var = None
        #---Choose the best varible in the rest var_pool--------
        for i in var_pool:
            # -------Put the new varible and the current varible together to train the model------
            cur_x = X[:,select_fetures_idx+[i]]           # let the current and new varible be the model data                
            clf.fit(cur_x,y)                        # Train the model
            pred_prob_y = clf.predict_proba(cur_x)[:,1]       # Predict the probality
            cur_auc = metrics.roc_auc_score(y,pred_prob_y)      # Calculate the AUC
            # ------Update the best varible---------------------------
            if(cur_auc>max_auc):
                max_auc =  cur_auc
                best_var = i
        #-------Check the effect of the new varible---------------------------
        last_auc = auc_rec[-1] if len(auc_rec)>0 else 0.0001
        valid = True  if ((max_auc-last_auc)/last_auc>0.005) else False
        
        # If the effect is remarkable, add the new varible into the chosen varible
        if(valid):
            print("The best AUC in this turn:",max_auc,",The best varible: ",feature_names[best_var])
            auc_rec.append(max_auc)  
            select_fetures_idx.append(best_var)  
            var_pool = var_pool[var_pool!=best_var]
        # If there is no remarkable effect, stop adding the varible
        else:
            print("The best AUC in this turn:",max_auc,",The best varible: ",feature_names[best_var],',No remarkable effect, stop addng')
            break
    return select_fetures_idx,feature_names[select_fetures_idx]

#----Data normalization------
xmin   = X.min(axis=0)
xmax   = X.max(axis=0)
X_norm =(X-xmin)/(xmax-xmin)
#Put the data into train data and test data
X_train,X_test,y_train,y_test = train_test_split(X_norm,y,test_size=0.2,random_state=0)

# Initialize the model
clf = LogisticRegression(penalty='l2',dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, 
                   class_weight=None, random_state=0, solver='lbfgs', max_iter=100, multi_class='ovr',
                   verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)

# ----Use stepwise regression----
select_fetures_idx,select_fetures = select_feture(clf,X_train,y_train,feature_names)
print("The final varible",len(select_fetures),":",select_fetures)

#----Use train data to train the model------
clf.fit(X_train[:,select_fetures_idx],y_train)


######################################
########## Model Evaluation ##########

y_score = clf.predict_proba(X_test[:,select_fetures_idx])[:,1]
fpr, tpr, threshold = roc_curve(y_test, y_score)
auc = roc_auc_score(y_test, y_score)

plt.figure(figsize = (10, 7))
plt.plot(fpr, tpr, label = "This Model")
plt.plot([0, 1], [0, 1], '--', label = "Random Classifier")
plt.xlabel("FPR (False Positive Rate)")
plt.ylabel("TPR (True Positive Rate)")
plt.title("ROC Curve")
plt.legend()
plt.show()
print("AUC: %.5f" % auc)

########## Score Transformation ##########

# xlsx = pd.ExcelWriter(r"C:\\Users\\apk15\\Desktop\data2.xlsx")

# Normalized data
w_norm = clf.coef_[0]
b_norm = clf.intercept_[0]
# Raw data
w = w_norm/(xmax[select_fetures_idx]-xmin[select_fetures_idx])
b = b_norm -  (w_norm/(xmax[select_fetures_idx] - xmin[select_fetures_idx])).dot(xmin[select_fetures_idx])
# Package
model_feature = feature_names[select_fetures_idx].to_list()
woe_tb =woe_tb_all[ woe_tb_all['feature'].isin(model_feature)].reset_index(drop=True)
md_logit ={ 'woe_tb':woe_tb,'model_feature':model_feature,'w':w,'b':b }

model_feature = md_logit['model_feature']
b            = md_logit['b']
woe_tb       = md_logit['woe_tb'].copy()


# Initialization of coefficients
init = 600
odds = 50
pdo = 10
# Calculation
factor = -pdo / math.log(2)
base_score = init - factor * (math.log(odds) - b)
score_tb = woe_tb.copy()
for i in range(len(model_feature)):
   score_tb.loc[score_tb['feature'] ==  model_feature[i],'coef'] = md_logit['w'][i]
score_tb['score'] = factor * score_tb['coef'] * score_tb['woe'] 
# score = base_score + feature_score
md = {'score_tb'   : score_tb,
      'base_score' : base_score,
      'model_feature'    : model_feature}

# score_tb.to_excel(xlsx, sheet_name="st1")
# score_tb.to_excel('C:\\Users\\apk15\\Desktop\data.xlsx')
print(base_score)
print(model_feature)
# print(md)

#
def cal_score(md,X):
    score_tb   = md['score_tb']
    base_score = md['base_score']
    features    = md['model_feature']
    
    # -----transform the table into dictionary----------
    score_dict ={feature:{}  for feature in features}
    for i in range(score_tb.shape[0]):
        feature = score_tb.loc[i,'feature']
        grp    = score_tb.loc[i,'grp']
        score_dict[feature][grp]= score_tb.loc[i,'score']
        
    # -----feature group to score -------- 
    score = X[features].copy()
    for col in features:    
        score[col]= score[col].map(score_dict[col])
        
    #-------final score---------------------------     
    score['base_score'] = base_score
    score['score'] = score.sum(axis=1)
    return score

#---Calculation------
model_X = raw_data[md['model_feature']]
pred_rs = cal_score(md,model_X)
pred_rs['is_bad'] = raw_data['is_bad']

# print(pred_rs)
# pred_rs.to_excel(xlsx, sheet_name="st2")

########## Determine Threshold ##########

def cal_score_threshold_tb(score_df,bin_step=10,is_bad_col_name='is_bad',score_col_name='score'):
    # ----- start and end --------------
    bin_start = math.trunc(score_df[score_col_name].min()/bin_step)*bin_step
    bin_end   =  math.trunc(score_df[score_col_name].max()/bin_step+1)*bin_step
    score_thd = pd.DataFrame(columns=['grp_names','clients','good_clients','bad_clients'])
    #----- numbers of good and bad clients -------
    for cur_bin in range(bin_start,bin_end,bin_step):
        cur_bin_name='['+str(cur_bin)+'-'+str(cur_bin+bin_step)+')'
        cur_score_df = score_df[(score_df[score_col_name]>=cur_bin)&(score_df[score_col_name]<cur_bin+bin_step)][is_bad_col_name]
        bad_cn = cur_score_df.sum()
        cn = cur_score_df.shape[0]
        score_thd.loc[score_thd.shape[0]]=[cur_bin_name,cn,cn-bad_cn,bad_cn]
      
    score_thd['sum_clients']            = score_thd['clients'].sum()
    score_thd['sum_good_clients']       = score_thd['good_clients'].sum()
    score_thd['sum_bad_clients']        = score_thd['bad_clients'].sum()
    score_thd['thershold']              = score_thd['grp_names'].apply(lambda x: '<'+x.split('-')[1].replace(')',''))  
    score_thd['lost_clients']           = score_thd['clients'].cumsum()
    score_thd['lost_clients%']          = score_thd['lost_clients']/score_thd['sum_clients']
    score_thd['lost_good_clients']      = score_thd['good_clients'].cumsum()
    score_thd['lost_good_clients%']     = score_thd['lost_good_clients']/score_thd['sum_good_clients']
    score_thd['lost_bad_clients']       = score_thd['bad_clients'].cumsum()
    score_thd['lost_bad_clients%']      = score_thd['lost_bad_clients']/score_thd['sum_bad_clients']
    tmp = score_thd['clients'].copy()
    tmp[tmp==0] = 1
    score_thd['bad_clients_percentage']         = score_thd['bad_clients']/tmp
    score_thd['bad_percentage_of_lost_clients'] = score_thd['lost_bad_clients']/score_thd['lost_clients']
    return score_thd

# -------- threshold table ---------------
score_df = pred_rs[['score','is_bad']]
score_thd=cal_score_threshold_tb(score_df,bin_step=10)

# score_thd.to_excel(xlsx, sheet_name="st3")

# xlsx.close()
# -------- score distribution ------------------
x_axis = score_thd['grp_names'].apply(lambda x: x.split('-')[0].replace('[',''))  
plt.bar(x_axis, score_thd['good_clients'], align="center", color="#66c2a5",label="good")
plt.bar(x_axis, score_thd['bad_clients'], align="center", bottom=score_thd['good_clients'], color='r', label="bad")
plt.rcParams["figure.figsize"] = (9, 4)
plt.legend()
plt.show()

###########################