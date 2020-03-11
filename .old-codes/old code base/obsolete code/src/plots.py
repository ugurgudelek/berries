
pred_class2 = pd.read_pickle("../result/predictions_model_class2_100epoch_2017_07_17 01_18_47_663062")
pred_class3 = pd.read_pickle("../result/predictions_model_class3_100epoch_2017_07_17 00_16_57_239094")
pred_regr = pd.read_pickle("../result/predictions_model_regr_100epoch_qratio_0_2017_07_16 20_25_16_566179")

# PR curves for regression and classification with 2 class
plt.figure()
prec_recall_curve_class2 = sklearn.metrics.precision_recall_curve(pred_class2['Act0'].values, pred_class2['Pred0'].values)
prec_recall_curve_reg2 = sklearn.metrics.precision_recall_curve(pred_regr['Actual'].values > 0, (pred_regr['Prediction'].values + 1) / 2)
plt.plot(prec_recall_curve_class2[1], prec_recall_curve_class2[0], '-', label = "Classification")
plt.plot(prec_recall_curve_reg2[1], prec_recall_curve_reg2[0], '--', label = "Regression")
#plt.title("PR Curve for Classification and Regression(Quantized)")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc = "upper right")
plt.show()

# PR curves for regression and classification with 3 class
plt.figure()
prec_recall_curve_class3_1 = sklearn.metrics.precision_recall_curve(pred_class3['Act0'].values, pred_class3['Pred0'].values)
prec_recall_curve_class3_2 = sklearn.metrics.precision_recall_curve(pred_class3['Act1'].values, pred_class3['Pred1'].values)
prec_recall_curve_class3_3 = sklearn.metrics.precision_recall_curve(pred_class3['Act2'].values, pred_class3['Pred2'].values)
prec_recall_curve_reg3_1 = sklearn.metrics.precision_recall_curve(pred_regr['Actual'].values < -.38, 1 - ((pred_regr['Prediction'].values + 1) / 2))
prec_recall_curve_reg3_2 = sklearn.metrics.precision_recall_curve(np.logical_and(pred_regr['Actual'].values > -.38, pred_regr['Actual'].values < .38), 1 - abs(pred_regr['Prediction'].values))
prec_recall_curve_reg3_3 = sklearn.metrics.precision_recall_curve(pred_regr['Actual'].values > .38, (pred_regr['Prediction'].values + 1) / 2)
plt.plot(prec_recall_curve_class3_1[1], prec_recall_curve_class3_1[0], linestyle = '-', label = "1 vs all (c)")
plt.plot(prec_recall_curve_class3_2[1], prec_recall_curve_class3_2[0], linestyle = '--', label = "2 vs all (c)")
plt.plot(prec_recall_curve_class3_3[1], prec_recall_curve_class3_3[0], linestyle = '-.', label = "3 vs all (c)")
plt.plot(prec_recall_curve_reg3_1[1], prec_recall_curve_reg3_1[0], linestyle = '-', linewidth = 3, label = "1 vs all (r)")
plt.plot(prec_recall_curve_reg3_2[1], prec_recall_curve_reg3_2[0], linestyle = '--', linewidth = 3, label = "2 vs all (r)")
plt.plot(prec_recall_curve_reg3_3[1], prec_recall_curve_reg3_3[0], linestyle = '-.', linewidth = 3, label = "3 vs all (r)")
#plt.title("PR Curve for Classification and Regression(Quantized) with 3 Class")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc = "upper right")
plt.show()

# ROC curves for regression and classification with 2 class
plt.figure()
roc_curve_class2 = sklearn.metrics.roc_curve(pred_class2['Act0'].values, pred_class2['Pred0'].values)
roc_curve_reg2 = sklearn.metrics.roc_curve(pred_regr['Actual'].values > 0, (pred_regr['Prediction'].values + 1) / 2)
plt.plot(roc_curve_class2[0], roc_curve_class2[1], label = "Classification")
plt.plot(roc_curve_reg2[0], roc_curve_reg2[1], label = "Regression")
#plt.title("ROC Curve for Classification and Regression(Quantized)")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend(loc = "lower right")
plt.show()

# ROC curves for regression and classification with 3 class
plt.figure()
roc_curve_class3_1 = sklearn.metrics.roc_curve(pred_class3['Act0'].values, pred_class3['Pred0'].values)
roc_curve_class3_2 = sklearn.metrics.roc_curve(pred_class3['Act1'].values, pred_class3['Pred1'].values)
roc_curve_class3_3 = sklearn.metrics.roc_curve(pred_class3['Act2'].values, pred_class3['Pred2'].values)
roc_curve_reg3_1 = sklearn.metrics.roc_curve(pred_regr['Actual'].values < -.38, 1 - ((pred_regr['Prediction'].values + 1) / 2))
roc_curve_reg3_2 = sklearn.metrics.roc_curve(np.logical_and(pred_regr['Actual'].values > -.38, pred_regr['Actual'].values < .38), 1 - abs(pred_regr['Prediction'].values))
roc_curve_reg3_3 = sklearn.metrics.roc_curve(pred_regr['Actual'].values > .38, (pred_regr['Prediction'].values + 1) / 2)
plt.plot(roc_curve_class3_1[0], roc_curve_class3_1[1], linestyle = '-', label = "1 vs all (c)")
plt.plot(roc_curve_class3_2[0], roc_curve_class3_2[1], linestyle = '--', label = "2 vs all (c)")
plt.plot(roc_curve_class3_3[0], roc_curve_class3_3[1], linestyle = '-.', label = "3 vs all (c)")
plt.plot(roc_curve_reg3_1[0], roc_curve_reg3_1[1], linestyle = '-', linewidth = 3, label = "1 vs all (r)")
plt.plot(roc_curve_reg3_2[0], roc_curve_reg3_2[1], linestyle = '--', linewidth = 3, label = "2 vs all (r)")
plt.plot(roc_curve_reg3_3[0], roc_curve_reg3_3[1], linestyle = '-.', linewidth = 3, label = "3 vs all (r)")
#plt.title("ROC Curve for Classification and Regression(Quantized) with 3 Class")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend(loc = "lower right")
plt.show()

# confusion matrix for 2 class
conf_reg_2 = sklearn.metrics.confusion_matrix(pred_regr['Actual'].values > 0, pred_regr['Prediction'].values > 0)
df_conf_reg_2 = pd.DataFrame(conf_reg_2, index = [i for i in ['[-1,0]', '[0,1]']], columns = [i for i in ['[-1,0]', '[0,1]']])
plt.figure(figsize = (4,3))
sn.heatmap(df_conf_reg_2, annot = True)
plt.show()

a = np.zeros(pred_class2.shape[0])
b = np.zeros(pred_class2.shape[0])
a[pred_class2['Act0'] == 1] = 0
a[pred_class2['Act1'] == 1] = 1
b[pred_class2['Pred0'] > pred_class2['Pred1']] = 0 
b[pred_class2['Pred0'] < pred_class2['Pred1']] = 1 
conf_class_2 = sklearn.metrics.confusion_matrix(a,b)
df_conf_class_2 = pd.DataFrame(conf_class_2, index = [i for i in ['0', '1']], columns = [i for i in ['0', '1']])
plt.figure(figsize = (4,3))
sn.heatmap(df_conf_class_2, annot = True)
plt.show()

# confusion matrix for 3 class
a = np.zeros(pred_regr.shape[0])
b = np.zeros(pred_regr.shape[0])
a[pred_regr['Actual'] < -.38] = 0
a[np.logical_and(pred_regr['Actual'] > -.38, pred_regr['Actual'] < .38)] = 1
a[pred_regr['Actual'] > .38] = 2
b[pred_regr['Prediction'] < -.38] = 0
b[np.logical_and(pred_regr['Prediction'] > -.38, pred_regr['Prediction'] < .38)] = 1
b[pred_regr['Prediction'] > .38] = 2
conf_reg_3 = sklearn.metrics.confusion_matrix(a,b)
df_conf_reg_3 = pd.DataFrame(conf_reg_3, index = [i for i in ['[-1,-.38]', '[-.38 ,.38]', '[.38,1]']], columns = [i for i in ['[-1,-.38]', '[-.38 ,.38]', '[.38,1]']])
plt.figure(figsize = (4,3))
sn.heatmap(df_conf_reg_3, annot = True)
plt.show()

a = np.zeros(pred_class3.shape[0])
b = np.zeros(pred_class3.shape[0])
a[pred_class3['Act0'] == 1] = 0
a[pred_class3['Act1'] == 1] = 1
a[pred_class3['Act2'] == 1] = 2
b[np.argmax(np.concatenate((pred_class3['Pred0'].values.reshape(-1,1), pred_class3['Pred1'].values.reshape(-1,1), pred_class3['Pred2'].values.reshape(-1,1)), axis = 1), axis = 1) == 0] = 0 
b[np.argmax(np.concatenate((pred_class3['Pred0'].values.reshape(-1,1), pred_class3['Pred1'].values.reshape(-1,1), pred_class3['Pred2'].values.reshape(-1,1)), axis = 1), axis = 1) == 1] = 1
b[np.argmax(np.concatenate((pred_class3['Pred0'].values.reshape(-1,1), pred_class3['Pred1'].values.reshape(-1,1), pred_class3['Pred2'].values.reshape(-1,1)), axis = 1), axis = 1) == 2] = 2
conf_class_3 = sklearn.metrics.confusion_matrix(a,b)
df_conf_class_3 = pd.DataFrame(conf_class_3, index = [i for i in ['0', '1', '2']], columns = [i for i in ['0', '1', '2']])
plt.figure(figsize = (4,3))
sn.heatmap(df_conf_class_3, annot = True)
plt.show()







