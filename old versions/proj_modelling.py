from pandas import *
from matplotlib.pyplot import * 	
from numpy import *	
from pymysql import *
from scipy.stats import kruskal, chisquare
from scipy.sparse import csr_matrix
from seaborn import *
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier#, GradientBoostingClassifier
from sklearn import metrics
#from sklearn.tree import export_graphviz
from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBClassifier, plot_importance
#import pydot

## ~~~~~~ Data Cleaning Functions ~~~~~~ ##
def period_to_underscore(the_input):
	the_input.columns = [i.replace(".", "_") for i in the_input.columns]

	return the_input

## ~~~~~~ Model Preparation Functions ~~~~~~ ##
def cat_to_num(the_input):
	the_input_1 = the_input[["Outcome_Type"]]
	the_input_2 = the_input.drop(["Outcome_Type"], axis = 1)

	cols = ["Outcome_Subtype", "Sex_upon_Outcome", "Age_upon_Outcome",  "Outcome_Month", "Intake_Type", "Intake_Condition", "Found_Location", "Days_to_Outcome", "Purebred_or_Mixed", "Gender", "Gen_Color", "Gen_Group"] #["Outcome_Subtype", "Outcome_Month", "Intake_Type", "Intake_Condition", "Found_Location", "Purebred_or_Mixed", "AKC_Group", "Gender", "Sex_Difference", "Gen_Color", "Sex_upon_Outcome", "Gen_Group"]
	prefixes = ["", "", "", "", "", "", "", "", "", "", "", ""]
	prefix_seps = [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "]

	# cols = ["Outcome_Month", "Intake_Type", "Intake_Condition", "Found_Location", "Purebred_or_Mixed", "AKC_Group", "Gender", "Sex_Difference", "Gen_Color", "Sex_upon_Outcome", "Gen_Group"]
	# prefixes = ["", "", "", "", "", "", "", "Was", "", "", "Gen"]
	# prefix_seps = [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", "_"]

	the_input_2 = get_dummies(the_input_2, columns = cols, prefix = prefixes, prefix_sep = prefix_seps)

	return the_input_2, the_input_1

def balance_data(temp_feat, temp_lab):
	# Saving the Names of the DataFrame Headers.
	cols_feat = list(temp_feat)
	cols_lab = list(temp_lab)

	# Converting temp_feat into a Sparse Matrix to Apply Over Sampling.
	matrix_feat = csr_matrix(temp_feat.values)
	ovr_train = RandomOverSampler(sampling_strategy = "not majority")
	new_feat, new_lab = ovr_train.fit_sample(matrix_feat, temp_lab.values.ravel())

	# Convering the Results Back into a DataFrame.
	df_feat = DataFrame(new_feat.toarray())
	df_feat.columns = cols_feat
	df_lab = DataFrame(new_lab)
	df_lab.columns = cols_lab

	return df_feat, df_lab

## ~~~~~~ Modelling Functions ~~~~~~ ##
def random_forest(feat, lab, test_feat, test_lab, feature_plot):
	rf = RandomForestClassifier(n_estimators = 200, random_state = 136, max_depth = 6)
	rf.fit(feat, lab.values.ravel())
	tree_pred = rf.predict(test_feat)

	print("Accuracy = ", metrics.accuracy_score(test_lab, tree_pred))
	print("Precision = ", metrics.precision_score(test_lab, tree_pred, average = "macro"))
	print("Recall = ", metrics.recall_score(test_lab, tree_pred, average = "macro"))
	print("Confusion Matrix:")
	print(metrics.confusion_matrix(test_lab, tree_pred))
	
	if feature_plot == True:
		features = list(feat)
		importances = rf.feature_importances_
		indices = argsort(importances)[::-1]
		indices = indices[0:10][::-1]

		title('Feature Importances')
		barh(range(len(indices)), importances[indices], color='b', align='center')
		yticks(range(len(indices)), [features[i] for i in indices])
		xlabel('Relative Importance')
		show()

	# if tree_plot == True:
	# 	dot_data = export_graphviz(rf.estimators_[5], out_file = 'small_tree.dot', feature_names = list(feat), class_names = lab.Outcome_Type.unique(), rounded = True, precision = 1)
	# 	(graph, ) = pydot.graph_from_dot_file('small_tree.dot')
	# 	graph.write_png('small_tree.png')


## ~~~~~~ Main Program ~~~~~~ ##
to_model = read_csv("/Users/Diana/Documents/Capstone Project/dog_total.csv")
to_model = period_to_underscore(to_model)
to_model = to_model.drop(['Unnamed: 0'], axis = 1)
#to_model = to_model.drop(['Outcome_Subtype'], axis = 1)

features, labels = cat_to_num(to_model)

train_feat, test_feat, train_lab, test_lab = train_test_split(features, labels, test_size = 0.3, random_state = 136)

train_feat, train_lab = balance_data(train_feat, train_lab)

# xgb = XGBClassifier()
# xgb.fit(train_feat, train_lab.values.ravel())
# xgb_pred = xgb.predict(test_feat)

# print(metrics.accuracy_score(test_lab, xgb_pred))

# plot_importance(xgb, max_num_features = 10)
# show()

random_forest(train_feat, train_lab, test_feat, test_lab, False)



