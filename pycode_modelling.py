from pandas import * # [10]
from matplotlib.pyplot import *  # [5]
from numpy import * # [11]
from scipy.stats import shapiro, kruskal, chisquare # [6]
from scipy.sparse import csr_matrix # [6]
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, scale, normalize # [12]
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV # [12]
from sklearn.ensemble import RandomForestClassifier # [12]
from sklearn import metrics # [12]
from sklearn.tree import export_graphviz, DecisionTreeClassifier # [12]
from sklearn.naive_bayes import MultinomialNB # [12]
from sklearn.neighbors import KNeighborsClassifier # [12]
from imblearn.over_sampling import RandomOverSampler # [9]
from xgboost import XGBClassifier, plot_importance # [3]
from sklearn.neural_network import MLPClassifier # [12]
from eli5 import format_as_dataframe # [8]
from eli5.sklearn.explain_weights import * # [8]
from eli5.sklearn.explain_prediction import * # [8]
from eli5.xgboost import * # [8]
import pydot
from sklearn.utils.multiclass import unique_labels # [12]
from datetime import datetime

startTime = datetime.now()
random.seed(136)

## ~~~~~~ Data Cleaning Functions ~~~~~~ ##
def period_to_underscore(the_input):
	the_input.columns = [i.replace(".", "_") for i in the_input.columns]

	return the_input

## ~~~~~~ Model Preparation Functions ~~~~~~ ##
def cat_to_num(the_input):
	the_input_1 = the_input[["Outcome_Type"]]
	the_input_2 = the_input.drop(["Outcome_Type"], axis = 1)
	the_input_3 = the_input_2.drop(["Age_upon_Outcome", "Days_to_Outcome", "Age_upon_Intake"], axis = 1)

	cols = list(the_input_3)
	prefixes = ["", "", "", "", "", "", "", "", "", ""]
	prefix_seps = [" ", " ", " ", " ", " ", " ", " ", " ", " ", " "]

	the_input_2 = get_dummies(the_input_2, columns = cols, prefix = prefixes, prefix_sep = prefix_seps)

	return the_input_2, the_input_1

def scale_data(the_input, standardize):
	if standardize == True:
		the_input[["Age_upon_Outcome", "Days_to_Outcome", "Age_upon_Intake"]] = scale(the_input[["Age_upon_Outcome", "Days_to_Outcome", "Age_upon_Intake"]])

	else:
		the_input[["Age_upon_Outcome", "Days_to_Outcome", "Age_upon_Intake"]] = normalize(the_input[["Age_upon_Outcome", "Days_to_Outcome", "Age_upon_Intake"]])

	return the_input

def balance_data(temp_feat, temp_lab):
	# Saving the Names of the DataFrame Headers.
	cols_feat = list(temp_feat)
	cols_lab = list(temp_lab)

	# Converting temp_feat into a Sparse Matrix to Apply Over Sampling.
	matrix_feat = csr_matrix(temp_feat.values)
	ovr_train = RandomOverSampler(sampling_strategy = "all")
	new_feat, new_lab = ovr_train.fit_sample(matrix_feat, temp_lab.values.ravel())

	# Convering the Results Back into a DataFrame.
	df_feat = DataFrame(new_feat.toarray())
	df_feat.columns = cols_feat
	df_lab = DataFrame(new_lab)
	df_lab.columns = cols_lab

	return df_feat, df_lab

# [6], [9], [10], [11], [12]

## ~~~~~~ Hyperparameter Tuning ~~~~~~ ##
def param_tuning(model, parameters, features, labels):
	feat, test_feat, lab, test_lab = train_test_split(features, labels, test_size = 0.3, random_state = 136, stratify = labels)
	feat, lab = balance_data(feat, lab)

	scorer = metrics.make_scorer(metrics.fbeta_score, beta = 1, average = "micro") # [4]

	grid = GridSearchCV(model, parameters, scoring = scorer, cv = 2) # [4]
	fit = grid.fit(feat, lab.values.ravel())

	print(fit.best_estimator_)

def best_k(train_feat, train_lab, test_feat, test_lab, findk):
	if findk == True:
		k = array([120,130,140])
		i = 0
		f_score = []

		for j in k:
			f1 = knn_model(train_feat, train_lab, test_feat, test_lab, False, False, k[i])
			
			f_score.append(f1)

			print("Completed Loop ", j)
			print(datetime.now() - startTime)
			print("")

			i += 1

		plot(k, f, 'go-')
		xlabel("Value of k")
		ylabel("F1 Score (Weighted)")
		title("F1 Score vs Value of k")
		show()

# [5], [9], [10], [11], [12]

## ~~~~~~ Modelling Functions ~~~~~~ ##
def baseline(the_input):
	accuracy = len(to_model[to_model.Outcome_Type == "Adoption"])/the_input.shape[0]

	return accuracy

def prediction(model, test_feat, test_lab, cross_valid, name, pred_prob):
	if pred_prob == False:
		pred = model.predict(test_feat)

		if cross_valid == False:
			return results(test_lab, pred, cross_valid, name)

		else:
			performance =  results(test_lab, pred, cross_valid, name)
			return performance

	else:
		pred = model.predict(test_feat)
		results(test_lab, pred, cross_valid, name)

		if name == "Decision Tree" or name == "Random Forest":
			expl = explain_prediction_tree_classifier(model, test_feat.iloc[0])
			expl_df = format_as_dataframe(expl)
			
			if name == "Decision Tree":
				expl_df.to_csv('/.../expl_dt.csv')

			else:
				expl_df.to_csv('/.../expl_rf.csv')
				expfi = explain_rf_feature_importance(model, feature_names = list(test_feat))
				expfi_df = format_as_dataframe(expfi)
				expfi_df.to_csv('/.../expfi_rf.csv')

		elif name == "XGBoost":
			expl = explain_prediction_xgboost(model, test_feat.iloc[0])
			expl_df = format_as_dataframe(expl)
			expl_df.to_csv('/.../expl_xgb.csv')

			expfi = explain_weights_xgboost(model, feature_names = list(test_feat))
			expfi_df = format_as_dataframe(expfi)
			expfi_df.to_csv('/.../expfi_xgb.csv')

		else:
			print("//")

def results(lab, pred, cross_valid, model):

	if cross_valid == False:
		print(model, ":")
		print("Accuracy = ", metrics.accuracy_score(lab, pred))
		print("Cohen's Kappa = ", metrics.cohen_kappa_score(lab, pred))
		print("Confusion Matrix:")
		print(metrics.confusion_matrix(lab, pred))
		print(metrics.classification_report(lab, pred, target_names = unique_labels(lab), digits = 6))
		print("")

	else:
		metric = array([metrics.accuracy_score(lab, pred), metrics.precision_score(lab, pred, average = "weighted"), metrics.recall_score(lab, pred, average = "weighted"), metrics.f1_score(lab, pred, average = "weighted"), metrics.cohen_kappa_score(lab, pred)])
				
		return metric

def decision_tree(feat, lab, test_feat, test_lab, feature_plot, tree_plot, cross_valid, pred_prob):
	dt = DecisionTreeClassifier(random_state = 136, max_depth = 6)
	dt.fit(feat, lab.values.ravel())
	
	if feature_plot == True:
		features = list(feat)
		importances = dt.feature_importances_
		indices = argsort(importances)[::-1]
		indices = indices[0:10][::-1]

		title('Feature Importances for Decision Tree')
		barh(range(len(indices)), importances[indices], color='r', align='center')
		yticks(range(len(indices)), [features[i] for i in indices])
		xlabel('Relative Importance')
		show()

	if tree_plot == True:
		dot_data = export_graphviz(dt, out_file = 'decision_tree.dot', feature_names = list(feat), class_names = lab.Outcome_Type.unique(), rounded = True, precision = 1) # [7]
		(graph, ) = pydot.graph_from_dot_file('decision_tree.dot') # [7]
		graph.write_png('decision_tree.png') # [7]

	return prediction(dt, test_feat, test_lab, cross_valid, "Decision Tree", pred_prob)

def random_forest(feat, lab, test_feat, test_lab, feature_plot, tree_plot, cross_valid, pred_prob):
	rf = RandomForestClassifier(n_estimators = 500, random_state = 136, max_features = 10)
	rf.fit(feat, lab.values.ravel())
	
	if feature_plot == True:
		features = list(feat)
		importances = rf.feature_importances_
		indices = argsort(importances)[::-1]
		indices = indices[0:10][::-1]

		title('Feature Importances')
		barh(range(len(indices)), importances[indices], color='g', align='center')
		yticks(range(len(indices)), [features[i] for i in indices])
		xlabel('Relative Importance')
		show()

	return prediction(rf, test_feat, test_lab, cross_valid, "Random Forest", pred_prob)

def xgboost_model(feat, lab, test_feat, test_lab, feature_plot, cross_valid, pred_prob):
	xgb = XGBClassifier(max_depth = 8, n_estimators = 200, random_state = 136) # [2]
	xgb.fit(feat, lab.values.ravel()) # [2]

	if feature_plot == True:
		plot_importance(xgb, max_num_features = 10)
		show()

	return prediction(xgb, test_feat, test_lab, cross_valid, "XGBoost", pred_prob)

def knn_model(feat, lab, test_feat, test_lab, cross_valid, pred_prob):
	knn = KNeighborsClassifier(n_neighbors=114, weights = 'distance')
	knn.fit(feat, lab.values.ravel())

	return prediction(knn, test_feat, test_lab, cross_valid, "K-Nearest Neighbors", pred_prob)

def nn_model(feat, lab, test_feat, test_lab, cross_valid, pred_prob):
	mlp = MLPClassifier(activation = "logistic", hidden_layer_sizes=(14,), solver = "lbfgs", random_state = 136) # [13]
	mlp.fit(feat, lab.values.ravel()) # [13]

	return prediction(mlp, test_feat, test_lab, cross_valid, "Neural Network", pred_prob)

def total_model(feat, lab, test_feat, test_lab, feature_plot, tree_plot, cross_valid):
	decision_tree(feat, lab, test_feat, test_lab, feature_plot, tree_plot, cross_valid, True)
	random_forest(feat, lab, test_feat, test_lab, feature_plot, False, cross_valid, True)
	xgboost_model(feat, lab, test_feat, test_lab, feature_plot, cross_valid, True)
	nn_model(feat, lab, test_feat, test_lab, cross_valid, True)
	knn_model(feat, lab, test_feat, test_lab, cross_valid, True)

def cross_valid_fn(features, labels, cross_valid):
	if cross_valid == False:
		train_feat, test_feat, train_lab, test_lab = train_test_split(features, labels, test_size = 0.2, random_state = 136, stratify = labels)
		train_feat, train_lab = balance_data(train_feat, train_lab) # [1]

		total_model(train_feat, train_lab, test_feat, test_lab, False, False, cross_valid)

	else:
		performance_dt = DataFrame(columns = ['Accuracy', 'Precision', 'Recall', 'F1', 'Cohen_Kappa'])
		performance_rf = DataFrame(columns = ['Accuracy', 'Precision', 'Recall', 'F1', 'Cohen_Kappa'])
		performance_xgb = DataFrame(columns = ['Accuracy', 'Precision', 'Recall', 'F1', 'Cohen_Kappa'])
		performance_nn = DataFrame(columns = ['Accuracy', 'Precision', 'Recall', 'F1', 'Cohen_Kappa'])
		performance_kn = DataFrame(columns = ['Accuracy', 'Precision', 'Recall', 'F1', 'Cohen_Kappa'])

		features, test_feats, labels, test_labs = train_test_split(features, labels, test_size = 0.2, random_state = 136, stratify = labels)
		
		kf = StratifiedKFold(n_splits = 10, random_state = 136)
		i = 0

		for train_i, test_i in kf.split(features, labels):
			train_feat, test_feat = features.iloc[train_i], features.iloc[test_i]
			train_lab, test_lab = labels.iloc[train_i], labels.iloc[test_i]
			train_feat, train_lab = balance_data(train_feat, train_lab)

			performance_dt.loc[i] = decision_tree(train_feat, train_lab, test_feat, test_lab, False, False, cross_valid, False)
			performance_rf.loc[i] = random_forest(train_feat, train_lab, test_feat, test_lab, False, False, cross_valid, False)
			performance_xgb.loc[i] = xgboost_model(train_feat, train_lab, test_feat, test_lab, False, cross_valid, False)
			performance_nn.loc[i] = nn_model(train_feat, train_lab, test_feat, test_lab, cross_valid, False)
			performance_kn.loc[i] = knn_model(train_feat, train_lab, test_feat, test_lab, cross_valid, False)

			i +=1

			print("Completed loop ", i)
			print(datetime.now() - startTime)

		print("Decision Tree: ")
		print(performance_dt.mean(axis = 0))
		print("")
		performance_dt.to_csv('/.../performance_dt.csv')

		print("Random Forest: ")
		print(performance_rf.mean(axis=0))
		print("")
		performance_rf.to_csv('/.../performance_rf.csv')

		print("XGBoost: ")
		print(performance_xgb.mean(axis=0))
		print("")
		performance_xgb.to_csv('/.../performance_xgb.csv')

		print("Neural Network: ")
		print(performance_nn.mean(axis=0))
		print("")
		performance_nn.to_csv('/.../performance_nn.csv')

		print("K-Nearest Neighbors: ")
		print(performance_kn.mean(axis=0))
		print("")
		performance_kn.to_csv('/.../performance_knn.csv')

		print(datetime.now() - startTime)
		print("")

		print("Using Test Set: ")
		total_model(train_feat, train_lab, test_feats, test_labs, True, True, False)

# [3], [5], [8], [10], [11], [12]

## ~~~~~~ Main Program ~~~~~~ ##
to_model = read_csv("/.../animal_total.csv")
to_model = period_to_underscore(to_model)
to_model = to_model.drop(['Unnamed: 0'], axis = 1)

print(baseline(to_model))

features, labels = cat_to_num(to_model)

features = scale_data(features, True)

hyperparameter_tuning = False

if hyperparameter_tuning == True:
	parameters = {'hidden_layer_sizes': [(13,13,13), (20,20,20), (50,50,50), (10,10,10), (14,14,14)]} # [4]
	mod = MLPClassifier(solver = 'lbfgs', activation = 'logistic', random_state = 136) # [4]
	param_tuning(mod, parameters, features, labels)

	train_feat, test_feat, train_lab, test_lab = train_test_split(features, labels, test_size = 0.2, random_state = 136, stratify = labels)
	best_k(train_feat, test_feat, train_lab, test_lab, hyperparameter_tuning)

cross_valid_fn(features, labels, True)

print(datetime.now() - startTime)

# [10], [12]

## ~~~~~~ Works Cited ~~~~~~ ##
# [1] Altini, Marco. “Dealing with Imbalanced Data: Undersampling, Oversampling and Proper Cross-Validation.” MarcoAltini.com. https://www.marcoaltini.com/blog/dealing-with-imbalanced-data-undersampling-oversampling-and-proper-cross-validation.
# [2] Brownlee, Jason. “How to Develop Your First XGBoost Model in Python with scikit-learn.” Machine Learning Mastery. https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/.
# [3] Chen, Tianqi, and Carlos Guestrin. "Xgboost: A scalable tree boosting system." In Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining, pp. 785-794. ACM, 2016.
# [4] Hariharan, Ashwin. How to use Machine Learning to Predict the Quality of Wines. Last Modified February 7, 2018. https://medium.freecodecamp.org/using-machine-learning-to-predict-the-quality-of-wines-9e2e13d7480d.
# [5] Hunter, John D. "Matplotlib: A 2D graphics environment." Computing in science & engineering 9, no. 3 (2007): 90-95.
# [6] Jones, Eric, Travis Oliphant, and Pearu Peterson. "{SciPy}: open source scientific tools for {Python}." (2014).
# [7] Koehrsen, William. “How to Visualize a Decision Tree from a Random Forest in Python using Scikit-Learn.” Towards Data Science. https://towardsdatascience.com/how-to-visualize-a-decision-tree-from-a-random-forest-in-python-using-scikit-learn-38ad2d75f21c.
# [8] Korobov, Mikhail, and Konstantin Lopuhin. “ELI5 Documentation. Last Modified November 19, 2018. https://media.readthedocs.org/pdf/eli5/latest/eli5.pdf.
# [9] Lemaître, Guillaume, Fernando Nogueira, and Christos K. Aridas. "Imbalanced-learn: A python toolbox to tackle the curse of imbalanced datasets in machine learning." The Journal of Machine Learning Research 18, no. 1 (2017): 559-563.
# [10] McKinney, Wes. "Data structures for statistical computing in python." In Proceedings of the 9th Python in Science Conference, vol. 445, pp. 51-56. 2010.
# [11] Oliphant, Travis E. A guide to NumPy. Vol. 1. USA: Trelgol Publishing, 2006.
# [12] Pedregosa, Fabian, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel et al. "Scikit-learn: Machine learning in Python." Journal of machine learning research 12, no. Oct (2011): 2825-2830.
# [13] Portilla, Jose. “A Beginner’s Guide to Neural Networks in Python and SciKit Learn 0.18.” Springboard Blog. https://www.springboard.com/blog/beginners-guide-neural-network-in-python-scikit-learn-0-18/.
