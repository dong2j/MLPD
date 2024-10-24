###                              CODE  FOR THE FOLLOWING MANUSCRIPT                                     ###
###                              PART II -- PHASE DIARAM PREDICTION                                     ###
#  TITLE: "Nonlinearity of the post-spinel transition and its expression in slabs and plumes worldwide"   #
#  AUTHORS: Junjie Dong, Rebecca A. Fischer, Lars P. Stixrude, Matthew C. Brennan, Kierstin Daviau,       #
#  Terry-Ann Suer, Katlyn M. Turner, Yue Meng, and Vitali B. Prakapenka                                   #

###                        MAIN PACKAGES USED FOR THIS CODE ARE LISTED AS FOLLOWS                       ###
# matplotlib=3.5.0                                                                                        #
# numpy=1.19.5                                                                                            #
# pandas=1.3.5                                                                                            #
# python=3.9.20                                                                                           #
# tqdm=4.66.5                                                                                             #
# scikit-learn=0.23.2                                                                                     #
# scipy=1.10.0                                                                                            #
### ALL THE PACKAGES INSTALLED IN THE CONDA ENVIRONMENT IN WHICH THE CODE WAS EXECUTED CAN BE FOUND IN  ###
# environment.yml                                                                                         # requirements.txt

### IMPORT REQUIRED PACKAGES ###

### NUMERICS AND DATA ###
import numpy as np
import pandas as pd

### REMOVE SOME WARNINGS ###
import warnings
warnings.filterwarnings('ignore')

### SKLEARN FUNCTIONS ###
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

### TIMEKEEPING ###
from tqdm import tqdm 

### FONTS, COLORS, STYLE FOR PLOTTING ###
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec
rcParams['font.family'] = 'arial'
cm_scale = 1/2.54
from matplotlib import ticker, cm
wd_c = ['#d1495b']
rw_c = ['#00798c']
pv_c = ['#edae49']
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
newcolors = np.array(['#d1495b',
                      '#00798c',
                      '#edae49'
])
phase_colors= ListedColormap(newcolors, name='mg2sio4')
ms = 10
lw = 1
lw_old = 1
ac_old = 1

### LOAD DATA FROM THIS WORK ###
pd_data_d1 = pd.read_csv("data/new/d1-kcl-phase.csv",header=0)
pd_data_d4 = pd.read_csv("data/new/d4-kcl-phase.csv",header=0)
pd_data_d11 = pd.read_csv("data/new/d11-kcl-phase.csv",header=0)
pd_data_d12 = pd.read_csv("data/new/d12-kcl-phase.csv",header=0)

pd_data_d2_s1 = pd.read_csv("data/new/d2-s1-kcl-phase.csv",header=0)
pd_data_d2_s2 = pd.read_csv("data/new/d2-s2-kcl-phase.csv",header=0)
pd_data_d2_s3 = pd.read_csv("data/new/d2-s3-kcl-phase.csv",header=0)
pd_data_d5_s1 = pd.read_csv("data/new/d5-s1-kcl-phase.csv",header=0)
pd_data_d5_s2 = pd.read_csv("data/new/d5-s2-kcl-phase.csv",header=0)
pd_data_d6_s1 = pd.read_csv("data/new/d6-s1-kcl-phase.csv",header=0)
pd_data_d6_s2 = pd.read_csv("data/new/d6-s2-kcl-phase.csv",header=0)
pd_data_d10_s1 = pd.read_csv("data/new/d10-s1-kcl-phase.csv",header=0)
pd_data_d10_s2 = pd.read_csv("data/new/d10-s2-kcl-phase.csv",header=0)
pd_data_d10_s3 = pd.read_csv("data/new/d10-s3-kcl-phase.csv",header=0)
pd_data_d13_s1 = pd.read_csv("data/new/d13-s1-kcl-phase.csv",header=0)
pd_data_d13_s2 = pd.read_csv("data/new/d13-s2-KCL-phase.csv",header=0)
pd_data_d14_s1 = pd.read_csv("data/new/d14-s1-kcl-phase.csv",header=0)
pd_data_d14_s2 = pd.read_csv("data/new/d14-s2-kcl-phase.csv",header=0)
pd_data_d14_s3 = pd.read_csv("data/new/d14-s2-kcl-phase.csv",header=0)

### COMPILE THEM INTO DATAFRAME ###
pb_data_dong_0 =  pd.concat([pd_data_d1,
                             pd_data_d2_s2,
                             pd_data_d2_s3,
                             pd_data_d4,
                             pd_data_d5_s1,
                             pd_data_d5_s2,
                             pd_data_d6_s2,
                             pd_data_d10_s1,
                             pd_data_d10_s2,
                             pd_data_d10_s3,
                             pd_data_d11,
                             pd_data_d13_s1,
                             pd_data_d13_s2,
                             pd_data_d14_s1,
                             pd_data_d14_s2,
                             pd_data_d14_s3
                           ])
###    REVISE COLUMN NAMES   ###
pb_data_dong =pb_data_dong_0[['p_kcl_corrected','temp','phase_num']].rename(columns={'p_kcl_corrected':'P','temp':'T','phase_num':'Phase'})

### LOAD SELECT DATA FROM LITERATURE ###
pb_data_k04 = pd.read_csv("data/literature/data-mg2sio4-k04-rw-corrected.csv",header=0)
pb_data_k09_wd = pd.read_csv("data/literature/data-mg2sio4-k09-wd-corrected.csv",header=0)
pb_data_i06 = pd.read_csv("data/literature/data-mg2sio4-i06-corrected.csv",header=0)
pb_data_c22 = pd.read_csv("data/literature/data-mg2sio4-c22-corrected.csv",header=0)

### COMPLIE DATA INTO A GLOBAL SET ###
frames= [pb_data_i06,pb_data_k04,pb_data_k09_wd ,pb_data_c22, pb_data_dong]
pb_data_0 = pd.concat(frames)
### FILTER DATA OUTSIDE THE P-T OF INTEREST ###
pb_data = pb_data_0[(pb_data_0['T']>=1573)&(pb_data_0['T']<=2673)&(pb_data_0['P']>=16)&(pb_data_0['P']<=28)]

### DATA PREPARATION ###
X = pb_data[['P','T']]
degree = 6 
#==> !!! DEGREE OF POLYNOMIALS USED, CHANGE THE VALUE BASED ON THE RESULTS OF MODEL SELECTION !!!
X_poly = PolynomialFeatures(degree=degree,include_bias=False).fit_transform(X)
Y = pb_data['Phase']
### DATA PREPARATION ###
model = LogisticRegression(C =0.01,class_weight = None,
                           multi_class = 'ovr', penalty = 'l1', 
                           solver = 'liblinear', random_state = 2022, max_iter = 500, tol = 1e-4) 
# ==> !!! FLAGS USED, CHANGE THE FLAGS BASED ON THE RESULTS OF MODEL SELECTION !!!
# Warnings about convergence failure may occur. 
# Increasing max_iter can resolve these but will also significantly increase computation time. 
# The phase diagram topology remains largely unchanged despite convergence issues
# so max_iter is kept low (500) to maintain efficiency.

modelfit = model.fit(X_poly, Y)

### CREATE MESH GRID OF THE P-T SPACE OF INTEREST###
pv, tv = np.meshgrid(np.linspace(14, 28, 500,endpoint=True),
                     np.linspace(1273, 2700, 500,endpoint=True))
X_pred = np.c_[pv.ravel(), tv.ravel()]
X_pred_poly = PolynomialFeatures(degree=degree,include_bias=False).fit_transform(X_pred)
### PREDICTING THE PHASE DIAGRAM FROM THE FITTED MODEL ###
Y_hat = modelfit.predict(X_pred_poly)
Y_proba_hat = modelfit.predict_proba(X_pred_poly)

### PLOTTING ###
fig = plt.figure(figsize=(8*cm_scale, 6*cm_scale),dpi=300)

plt.pcolormesh(pv, tv, Y_hat.reshape(pv.shape), cmap=phase_colors,alpha = 1)
plt.contour(pv, tv, Y_hat.reshape(pv.shape), levels =[0.000001,1,2,3], colors = 'k',alpha = 1, linestyles = '-')
plt.scatter( pb_data_0.values[:, 0],pb_data_0.values[:, 1],c=pb_data_0.values[:, 2]
           , edgecolors='k', s = ms, lw = 1,cmap=phase_colors, marker = 's',alpha = ac_old)

### CONTROL STYLE ###
plt.xlim(16, 28)
plt.ylim(1573, 2523)
plt.tick_params(labelsize=8)
plt.minorticks_on()
plt.tick_params(direction='in')
plt.tick_params(direction='in',which="minor")
plt.xlabel('Pressure (GPa)',fontsize=8, labelpad=5)
plt.ylabel('Temperature (K)',fontsize=8, labelpad=5)

### SAVE THE FIGURE ###
fig.savefig('logit_reg_fit.pdf', bbox_inches = "tight")
plt.show()
