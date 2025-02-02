# import all relevant modules

import pandas as pd
import numpy as np
import scipy.io as sp
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import shuffle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from scipy.stats import ttest_ind
from scipy.stats import ttest_1samp
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA 
from scipy.stats import ttest_rel
from sklearn.utils import resample

# function to process the mat file for downstream processing

def gcparser(mat):
    data = np.transpose(mat['XTIC'])  #XTIC is a matix of measured values from GC-MS
    sample_names = mat['SAM']  # SAM contains the name of each sample
    sample_names = np.hstack(np.hstack(sample_names)).tolist()  # convert nested numpy arrays into a list
    RT = mat['RT']  # RT is retention time (in minutes)
    RT = np.hstack(np.hstack(RT)).tolist()  # convert nested numpy arrays into a list
    y = mat['CLASS']  #CLASS is the diagnosis of each sample (in this casse 1=control; 2=CD)
    y = np.hstack(y).tolist()  # convert nested numpy arrays into a list
    # put pieces back together in a pandas dataframe
    return pd.DataFrame(data, columns=sample_names, index=RT)

# preparing the file by creating x and y values to be used in models

def modelprep(file):
    mat_file = sp.loadmat(file)
    df = gcparser(mat_file)
    y_df = df.copy()
    y = y_df.columns.str.split("_").str[2]
    x = df.T
    return x, y

def file_display(file):
    mat_file = sp.loadmat(file)
    df = gcparser(mat_file)
    return df.describe()
    return df.head()

# plot of ion counts

def ion_plot(file):
    mat_file = sp.loadmat(file)
    df = gcparser(mat_file)
    df.plot(kind='box', figsize=(12, 8))
    plt.title(f'Ion Count Distribution for Each Sample in {file_dict2.get(file)}')
    plt.ylabel('Ion Count')
    plt.xticks(rotation='vertical')
    plt.show()

# confusion matrix of svm model

def c_svm_matrix(x,y):
    clf = svm.SVC()
    clf.fit(x,y)
    predictions = clf.predict(x)
    
    cm = confusion_matrix(y, predictions)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()

# confusion matrix of random forest model

def c_rf_matrix(x,y):
    rf = RandomForestClassifier()
    rf.fit(x,y)
    predictions = rf.predict(x)
    
    cm = confusion_matrix(y, predictions)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()

# function of svm model

def svm_modeltest(x,y):
    # create empty lists to store all my for loop data
    acc_list = []
    specificity_list = []
    sensitivity_list = [] 
    f1_list = []
    roc_auc_list_rand = []
    roc_auc_list = []

    # first bootstrap to test on random chance
    for i in range(0,300):
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3)
        clf = svm.SVC(probability=True)
        clf.fit(x_train, y_train)
        if len(np.unique(y_test)) > 1:  # Only calculate ROC AUC if both classes are present
            y_probs = clf.predict_proba(x_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_probs)
            roc_auc_list_rand.append(roc_auc)
        else:
            roc_auc_list_rand.append(np.nan)    


    # second bootstrap to test on permutation
    x, y = shuffle(x, y, random_state=42)
    
    for i in range(0,300):
        # create training data and test data
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state=42)

        # create accuracy score 
        clf = svm.SVC(probability=True)
        clf.fit(x_train,y_train)
        predictions = clf.predict(x_test)
        accuracy = accuracy_score(y_test,predictions)
        acc_list.append(accuracy)

        # create confusion matrix and work out specificity and sensitivity
        cm = confusion_matrix(y_test, predictions)
        sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) != 0 else 0
        sensitivity_list.append(sensitivity)
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) != 0 else 0
        specificity_list.append(specificity)

        # create f1 score
        f1 = f1_score(y_test, predictions, pos_label='CD', average='binary')
        f1_list.append(f1)

        # create roc-auc score
        y_probs = clf.predict_proba(x_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_probs)
        roc_auc_list.append(roc_auc)
    
    mean_acc = np.mean(acc_list)
    mean_specificity = np.mean(specificity_list)
    mean_sensitivity = np.mean(sensitivity_list)
    mean_f1 = np.mean(f1_list)
    mean_roc_auc = np.mean(roc_auc_list)
    mean_roc_auc_rand = np.mean(roc_auc_list_rand)

    # statistical test to compare permutation to random chance
    roc_auc_array = np.array(roc_auc_list)
    roc_auc_rand_array = np.array(roc_auc_list_rand)

    valid_ind = ~np.isnan(roc_auc_rand_array) 
    roc_auc_clean = roc_auc_array[valid_ind]
    roc_auc_rand_clean = roc_auc_rand_array[valid_ind]
    
    t_stat, p_value = ttest_rel(roc_auc_clean, roc_auc_rand_clean)

    return round(mean_acc, 3), round(mean_specificity, 3), round(mean_sensitivity, 3), round(mean_f1, 3), round(np.mean(roc_auc_clean), 3), round(np.mean(roc_auc_rand_clean), 3), p_value


# function of rf model

def rf_modeltest(x,y):
    # create empty lists to store all my for loop data
    acc_list = []
    specificity_list = []
    sensitivity_list = []
    f1_list = []
    roc_auc_list_rand = []
    roc_auc_list = []

    # first bootstrap to test on random chance
    for i in range(0,300):
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3)
        rf = RandomForestClassifier()
        rf.fit(x_train, y_train)
        if len(np.unique(y_test)) > 1:  # Only calculate ROC AUC if both classes are present
            y_probs = rf.predict_proba(x_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_probs)
            roc_auc_list_rand.append(roc_auc)
        else:
            roc_auc_list_rand.append(np.nan)

    # second bootstrap to test on permutation
    x, y = shuffle(x, y, random_state=42)

    for i in range(0,300):
        
        # create training data and test data
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state=42)

        # create accuracy score
        rf = RandomForestClassifier()
        rf.fit(x_train, y_train)
        predictions = rf.predict(x_test)
        accuracy = accuracy_score(y_test,predictions)
        acc_list.append(accuracy)

        # create confusion matrix and workout specificity and sensitivity
        cm = confusion_matrix(y_test, predictions)
        sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) != 0 else 0
        sensitivity_list.append(sensitivity)
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) != 0 else 0
        specificity_list.append(specificity)

        # create f1 score
        f1 = f1_score(y_test, predictions, pos_label='CD', average='binary')
        f1_list.append(f1)

        # create roc-auc score
        y_probs = rf.predict_proba(x_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_probs)
        roc_auc_list.append(roc_auc)
        
    mean_acc = np.mean(acc_list)
    mean_specificity = np.mean(specificity_list)
    mean_sensitivity = np.mean(sensitivity_list)
    mean_f1 = np.mean(f1_list)
    mean_roc_auc = np.mean(roc_auc_list)
    mean_roc_auc_rand = np.mean(roc_auc_list_rand)

    # statistical test
    roc_auc_array = np.array(roc_auc_list)
    roc_auc_rand_array = np.array(roc_auc_list_rand)

    # statistical test to compare permutation to random chance
    valid_ind = ~np.isnan(roc_auc_rand_array) 
    roc_auc_clean = roc_auc_array[valid_ind]
    roc_auc_rand_clean = roc_auc_rand_array[valid_ind]
    
    t_stat, p_value = ttest_rel(roc_auc_clean, roc_auc_rand_clean)

    return round(mean_acc, 3), round(mean_specificity, 3), round(mean_sensitivity, 3), round(mean_f1, 3), round(np.mean(roc_auc_clean), 3), round(np.mean(roc_auc_rand_clean), 3), p_value


# file retreival storage

file_dict = {
    'breath' : 'GCMS_data/breath/BWG_BR_CDvCTRL',
    'blood' : 'GCMS_data/blood/BWG_BL_CDvCTRL',
    'urine' : 'GCMS_data/urine/BWG_UR_CDvCTRL.mat',
    'faecal' : 'GCMS_data/faecal/BWG_FA_CDvCTRL.mat'    
}

file_dict2 = {v: k for k, v in file_dict.items()}

# original chromatography exploratory data analysis

def chroma_disp(file):
    mat_file = sp.loadmat(file)
    df = gcparser(mat_file)
    ion_count = df.index
    # y = df.columns.str.split("_").str[2]
    for c in df.columns:
        if c.endswith("CD"):
            df.rename(columns = {c : "CD"}, inplace=True)
        else:
            df.rename(columns = {c : "CTRL"}, inplace=True)
    for column in df.columns:
        plt.plot(ion_count, df[column])
    plt.xlabel('Retention Time (minutes)')
    plt.ylabel('Ion Count')
    plt.title(f'GC-MS Chromatogram of all {file_dict2.get(file)} samples')
    # plt.legend(loc="upper right")
    plt.show()

def chroma_disp(file):
    mat_file = sp.loadmat(file)
    df = gcparser(mat_file)
    ion_count = df.index
    # y = df.columns.str.split("_").str[2]
    for c in df.columns:
        if c.endswith("CD"):
            df.rename(columns = {c : "CD"}, inplace=True)
        else:
            df.rename(columns = {c : "CTRL"}, inplace=True)

    CTRL_columns = [col for col in df.columns if col.startswith("CTRL")]
    for column in CTRL_columns:
        plt.plot(ion_count, df[column], color="green")
    plt.xlabel('Retention Time (minutes)')
    plt.ylabel('Ion Count')
    plt.title(f'GC-MS Chromatogram of all {file_dict2.get(file)} samples')
    
    CD_columns = [col for col in df.columns if col.startswith("CD")]
    for column in CD_columns:
        plt.plot(ion_count, df[column], color = "red")
    plt.xlabel('Retention Time (minutes)')
    plt.ylabel('Ion Count')
    plt.title(f'GC-MS Chromatogram of all {file_dict2.get(file)} samples')

    plt.plot([], [], color="green", label="CTRL")  # Dummy entry for CTRL
    plt.plot([], [], color="red", label="CD")     # Dummy entry for CD
    plt.legend(loc="best") 
    plt.show()

def pca_disp(file):
    # parse the files
    mat_file = sp.loadmat(file)
    df = gcparser(mat_file)

    # declare the number of components and run PCA
    n_components = 2
    pca = PCA(n_components)
    pca.fit(df.T)
    X_scores = pd.DataFrame(pca.transform((df.T)), index=df.columns).T

    # change then names of the columns in dataframe to allow easier plotting
    for c in df.columns:
        if c.endswith("CD"):
            df.rename(columns = {c : "CD"}, inplace=True)
        else:
            df.rename(columns = {c : "CTRL"}, inplace=True)

    # set the colours used
    y = df.columns.str.split("_").str[2]
    colour_key = {'CTRL': 'green', 'CD': 'blue'}
    sample_colours = [colour_key[i] for i in df.columns]

    # plot the PCA
    plt.scatter(X_scores.iloc[0], X_scores.iloc[1], c = sample_colours)
    plt.xlabel('PC1 ('+str(round(pca.explained_variance_ratio_[0]*100))+'%)')
    plt.ylabel('PC2 ('+str(round(pca.explained_variance_ratio_[1]*100))+'%)')
    plt.title(f'Principal Component Analysis of {file_dict2.get(file)} samples')
    plt.scatter([], [], c='green', label='CTRL')  
    plt.scatter([], [], c='blue', label='Disease') 
    plt.legend(loc="upper center")
    plt.show()

for x in file_dict.values():
    pca_disp(x)

file_display("GCMS_data/breath/BWG_BR_CDvCTRL")
file_display("GCMS_data/blood/BWG_BL_CDvCTRL")
file_display("GCMS_data/faecal/BWG_FA_CDvCTRL.mat")
file_display("GCMS_data/urine/BWG_UR_CDvCTRL.mat")

# confusion matrix for breath sample
c_svm_matrix(*modelprep(file_dict.get("breath")))
c_rf_matrix(*modelprep(file_dict.get("breath")))
# confusion matrix for blood sample
c_svm_matrix(*modelprep(file_dict.get("blood")))
c_rf_matrix(*modelprep(file_dict.get("blood")))
# confusion matrix for urine sample
c_svm_matrix(*modelprep(file_dict.get("urine")))
c_rf_matrix(*modelprep(file_dict.get("urine")))
# confusion matrix for faecal sample
c_svm_matrix(*modelprep(file_dict.get("faecal")))
c_rf_matrix(*modelprep(file_dict.get("faecal")))

import pandas as pd

# Create an empty DataFrame with appropriate columns for final report
results_list = []
results_list2 = []

# Assuming file_dict is your dictionary and modelprep, svm_modeltest, rf_modeltest are predefined
for key, value in file_dict.items():
    # call the models
    svm_acc, svm_spec, svm_sens, svm_f1, svm_auroc_base, svm_auroc, svm_pvalue  = svm_modeltest(*modelprep(value))
    rf_acc, rf_spec, rf_sens, rf_f1, rf_auroc_base, rf_auroc, rf_pvalue  = rf_modeltest(*modelprep(value))
    
    # convert the necessary variables to percentages
    svm_acc, svm_spec, svm_sens, svm_f1 = [f"{x * 100:.2f}%" for x in[svm_acc, svm_spec, svm_sens, svm_f1]]
    rf_acc, rf_spec, rf_sens, rf_f1 = [f"{x * 100:.2f}%" for x in[rf_acc, rf_spec, rf_sens, rf_f1]]
    
    # add SVM results to the list 
    results_list.append([key, 'SVM', svm_acc, svm_spec, svm_sens, svm_f1, svm_auroc_base])
    results_list2.append([key, 'SVM', svm_auroc, svm_pvalue, ("Yes" if svm_pvalue < 0.05 else "No")])
    
    # add RF results to the list 
    results_list.append([key, 'RF', rf_acc, rf_spec, rf_sens, rf_f1, rf_auroc_base])
    results_list2.append([key, 'RF', rf_auroc, rf_pvalue, ("Yes" if rf_pvalue < 0.05 else "No")])

# Convert the final results list into a DataFrame
results_df = pd.DataFrame(results_list, columns=["Sample Type", "Model", "Accuracy (%)", "Specificity (%)", "Sensitivity (%)", "F1 Score (%)", "AUROC (%)"])
results_df2 = pd.DataFrame(results_list2, columns=["Sample Type", "Model", "Permuted AUROC (%)", "p-value", "SSD"])

print(results_df)
results_df3 = results_df2.iloc[-2:]
print(results_df3)


