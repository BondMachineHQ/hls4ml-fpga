import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
import math
import statistics
import matplotlib.pyplot as plt
import scikitplot as skplt
from matplotlib import pyplot
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

def compute_accuracy(true_y, output_y):
    return accuracy_score(np.argmax(true_y, axis=1), np.argmax(output_y, axis=1))

def compute_f1_score(true_y, output_y):
    return f1_score(np.argmax(true_y, axis=1), np.argmax(output_y, axis=1), average='macro')
    
def compute_precision(true_y, output_y):
    return precision_score(np.argmax(true_y, axis=1), np.argmax(output_y, axis=1), average='macro')

def compute_recall(true_y, output_y):
    return recall_score(np.argmax(true_y, axis=1), np.argmax(output_y, axis=1), average='macro')

def compute_sensitivity(true_y, output_y):
    tn, fp, fn, tp = confusion_matrix(np.argmax(true_y, axis=1), np.argmax(output_y, axis=1)).ravel()
    return tp / (fn+tp)

def compute_specificity(true_y, output_y):
    tn, fp, fn, tp = confusion_matrix(np.argmax(true_y, axis=1), np.argmax(output_y, axis=1)).ravel()
    return tn / (tn+fp)

def get_roc_curve(true_y, output_y, model_name):
    fpr, tpr, threshold = metrics.roc_curve(np.argmax(true_y, axis=1), np.argmax(output_y, axis=1))
    roc_auc = metrics.auc(fpr, tpr)

    # method I: plt
    import matplotlib.pyplot as plt
    plt.title('Receiver Operating Characteristic of '+model_name)
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
def load_classes(dataset):
    classes = np.load('datasets/'+dataset+'_classes.npy', allow_pickle=True)
    return classes

def read_bm_output(dataset):
    res = []
    with open('outputs/'+dataset+'/bm_output.txt') as f:
        for line in f.readlines():
            splitted = line.replace("[","").replace("]", "").replace("\n", "").replace(" ", "").split(",")
            res.append([float(splitted[0]), float(splitted[1])])
            
    return np.array(res)

def read_software_output(dataset):
    y_keras = np.load("datasets/"+dataset+'_y_keras.npy')
    return y_keras
    
def read_hls4ml_output(dataset):
    res = []
    with open('outputs/'+dataset+'/hls4ml_output.txt') as f:
        for line in f.readlines():
            splitted = line.replace("[","").replace("]", "").replace("\n", "").replace(" ", "").split(",")
            res.append([float(splitted[0]), float(splitted[1])])
            
    return np.array(res)

def load_real_label(dataset):
    y_test = np.load("datasets/"+dataset+'_y_test.npy')
    return y_test

dataset_name = "banknote-authentication"

hls4ml_output = read_hls4ml_output(dataset_name)
bm_output = read_bm_output(dataset_name)

output_len = len(bm_output)

keras_output = read_software_output(dataset_name)[:output_len]
y_test = load_real_label(dataset_name)[:output_len]

global_pct_errors_bm_o0 = []
global_pct_errors_hls4ml_o0 = []

global_pct_errors_bm_o1 = []
global_pct_errors_hls4ml_o1 = []

for i in range(0, len(keras_output)):
    #if np.argmax(keras_output[i]) == 1:
    print("software  \t\t\toutput -> ", keras_output[i],     "    classification: ", np.argmax(keras_output[i]))
    print("bondmachine  \t\t\toutput -> ", bm_output[i],        "    classification: ", np.argmax(bm_output[i]))
    print("hls4ml  \t\t\toutput -> ", hls4ml_output[i],    "    classification: ", np.argmax(hls4ml_output[i]))

    p0_keras_output = keras_output[i][0]
    p1_keras_output = keras_output[i][1]
    
    p0_bm_output = bm_output[i][0]
    p1_bm_output = bm_output[i][1]
    
    p0_hls4ml_output = hls4ml_output[i][0]
    p1_hls4ml_output = hls4ml_output[i][1]
    
    pct_error_bm_o0 = (abs(p0_bm_output - p0_keras_output)/p0_keras_output) * 100
    pct_error_bm_o1 = (abs(p1_bm_output - p1_keras_output)/p1_keras_output) * 100
    
    pct_error_hls4ml_o0 = (abs(p0_hls4ml_output - p0_keras_output)/p0_keras_output) * 100
    pct_error_hls4ml_o1 = (abs(p1_hls4ml_output - p1_keras_output)/p1_keras_output) * 100
    
    global_pct_errors_bm_o0.append((pct_error_bm_o0))
    global_pct_errors_hls4ml_o0.append((pct_error_hls4ml_o0))
    
    global_pct_errors_bm_o1.append((pct_error_bm_o1))
    global_pct_errors_hls4ml_o1.append((pct_error_hls4ml_o1))

    print("\n")

print(" Average percentage error classification output 0 for bondmachine ", statistics.mean(global_pct_errors_bm_o0))
print(" Average percentage error classification output 0 for hls4ml      ", statistics.mean(global_pct_errors_hls4ml_o0))
print(" Average percentage error classification output 1 for bondmachine ", statistics.mean(global_pct_errors_bm_o1))
print(" Average percentage error classification output 1 for hls4ml      ", statistics.mean(global_pct_errors_hls4ml_o1))

print("\n")
print("Accuracy keras model (software)     on real label: {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(keras_output, axis=1))))
print("Accuracy bondmachine ml on ebaz4205 on real label: {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(bm_output, axis=1))))
print("Accuracy hls4ml         on ebaz4205 on real label: {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(hls4ml_output, axis=1))))

classes = load_classes(dataset_name)

accuracy_keras = compute_accuracy(y_test, keras_output)
f1_score_keras = compute_f1_score(y_test, keras_output)
precision_keras = compute_precision(y_test, keras_output)
recall_keras = compute_recall(y_test, keras_output)
sensititivy_keras = compute_sensitivity(y_test, keras_output)
specificity_keras = compute_specificity(y_test, keras_output)

accuracy_bm = compute_accuracy(y_test, bm_output)
f1_score_bm = compute_f1_score(y_test, bm_output)
precision_bm = compute_precision(y_test, bm_output)
recall_bm = compute_recall(y_test, bm_output)
sensititivy_bm = compute_sensitivity(y_test, bm_output)
specificity_bm = compute_specificity(y_test, bm_output)

accuracy_hls4ml = compute_accuracy(y_test, hls4ml_output)
f1_score_hls4ml = compute_f1_score(y_test, hls4ml_output)
precision_hls4ml = compute_precision(y_test, hls4ml_output)
recall_hls4ml = compute_recall(y_test, hls4ml_output)
sensititivy_hls4ml = compute_sensitivity(y_test, hls4ml_output)
specificity_hls4ml = compute_specificity(y_test, hls4ml_output)

print("keras accuracy: ", accuracy_keras)
print("keras f1 score: ", f1_score_keras)
print("keras precision: ", precision_keras)
print("keras recall: ", recall_keras)
print("keras sensitivity: ", sensititivy_keras)
print("keras specificity: ", specificity_keras)
print("\n")
print("bm accuracy: ", accuracy_bm)
print("bm f1 score: ", f1_score_bm)
print("bm precision: ", precision_bm)
print("bm recall: ", recall_bm)
print("bm sensitivity: ", sensititivy_bm)
print("bm specificity: ", specificity_bm)
print("\n")
print("hls4ml accuracy: ", accuracy_hls4ml)
print("hls4ml f1 score: ", f1_score_hls4ml)
print("hls4ml precision: ", precision_hls4ml)
print("hls4ml recall: ", recall_hls4ml)
print("hls4ml sensitivity: ", sensititivy_hls4ml)
print("hls4ml specificity: ", specificity_hls4ml)

get_roc_curve(y_test, keras_output, "keras")
get_roc_curve(y_test, bm_output, "bondmachine")
get_roc_curve(y_test, hls4ml_output, "hls4ml")