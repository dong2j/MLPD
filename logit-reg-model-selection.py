###                              CODE FOR THE FOLLOWING MANUSCRIPT                                      ###
###                              PART I -- MODEL SELECTION/HYPERPARAMETER TUNING                        ###
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
# environment.yml
# requirements.txt      

### IMPORT REQUIRED PACKAGES ###

### NUMERICS AND DATA ###
import numpy as np
import pandas as pd

### REMOVE SOME WARNINGS ###
import warnings
warnings.filterwarnings('ignore')

### SKLEARN FUNCTIONS ###
import sklearn
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

### TIMEKEEPING ###
from tqdm import tqdm 

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
                             pd_data_d2_s1,
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

### LOAD DATA FROM LITERATURE ###
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
y = pb_data['Phase']
### TRAIN-TEST SPLIT ###
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2022)
### EMPTY DICTIONARIES TO SAVE RESULTS ###
X_train_poly_dict = {}
X_test_poly_dict = {}
l1_scores = {}
best_params = {}
best_scores = {}
best_train_scores = {}
best_degrees = {}
### RANGE OF HYPERPARAMETERS SEARCHED ###
degrees = [1,2,3,4,5,6,7,8,9,10]
grid_range = [{'penalty': ['l1']
         ,'class_weight': ['balanced', None]
         ,'C': [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]
         ,'solver':['liblinear']
         ,'multi_class':['ovr']}]
scores = ['micro']
tuned_params = [grid_range]

### GRID SEARCH/HYPERPARAMETER TUNING ###
output_file = "log.txt"
i = 0
# Open file to save results
with open(output_file, "w") as f:
    # OUTER LOOP OVER METRICS
    # ONLY F1_micro IS USED HERE
    for score in tqdm(scores, desc='Scores', position=0):
        best_scores[score] = []
        best_train_scores[score] = []
        best_params[score] = []
        best_degrees[score] = []
        # LOOP OVER HYPERPARAMETERS
        for grid in tqdm(tuned_params, desc='Grid Search', position=1, leave=False):
            # LOOP OVER DEGREES
            for degree in tqdm(degrees, desc='Degrees', position=2, leave=False):
                X_train_poly_dict[degree] = PolynomialFeatures(degree=degree, include_bias=False).fit_transform(X_train)
                X_test_poly_dict[degree] = PolynomialFeatures(degree=degree, include_bias=False).fit_transform(X_test)
                print("--------------------------")
                print("### DEGREE:", degree, "###")
                print("# TUNING HYPER-PARAMETERS FOR F1_%s" % score,'...\n')
                clf = GridSearchCV(
                    LogisticRegression(random_state=2022), grid, cv=3, scoring='f1_%s' % score
                )
                clf.fit(X_train_poly_dict[degree], y_train)
                # Warnings about convergence failure may occur. 
                # Increasing max_iter can resolve these but will also significantly increase computation time. 
                # The phase diagram topology remains largely unchanged despite convergence issues
                # so when making final phase diagram (PART II), max_iter is kept low (500) to maintain efficiency.
                print("BEST PARAMETERS SET FOUND ON TRAINING SET:\n", clf.best_params_)
                f.write(f"--------------------------\n")
                f.write(f"### DEGREE: {degree} ###\n")
                f.write("BEST PARAMETERS SET FOUND ON TRAINING SET:\n")
                f.write(f"{clf.best_params_}\n")
                f.flush()
                best_params[score].append(clf.best_params_)
                best_degrees[score].append(degree)
                y_true, y_pred = y_test, clf.predict(X_test_poly_dict[degree])
                y_train_pred = clf.predict(X_train_poly_dict[degree])
                test_score = f1_score(y_true, y_pred, average=score)
                train_score = f1_score(y_train.values, y_train_pred, average=score)
                print(f'F1 SCORE (TEST): {test_score}')
                print(f'F1 SCORE (TRAINING): {train_score}')
                f.write(f'F1 SCORE (TEST): {test_score}\n')
                f.write(f'F1 SCORE (TRAINING): {train_score}\n')
                f.flush()
                best_scores[score].append(test_score)
                best_train_scores[score].append(train_score)
                i += 1
    best_idx = np.array(best_scores[score]).argmax()
    
    print("--------------------------")
    print('!!! BEST DEGREE =', best_degrees[score][best_idx])
    print('!!! BEST PARAMETER SET =', best_params[score][best_idx])
    f.write(f'--------------------------\n')
    f.write(f'!!! BEST DEGREE = {best_degrees[score][best_idx]}\n')
    f.write(f'!!! BEST PARAMETER SET = {best_params[score][best_idx]}\n')
    f.flush()

print('MODEL SELECTION DONE, RESULTS CAN BE FOUND IN', output_file)