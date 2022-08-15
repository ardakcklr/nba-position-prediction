# importing necessary libraries
import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV



# Possible warnings are ignored
import warnings
warnings.filterwarnings('ignore')

# All the files that start with string 'nba' in the project file are joined
joined_files = os.path.join(r"C:\Users\rdkck\PycharmProjects\CIDP_Project2", "nba*.csv")

# The list of all the files that are joined
joined_list = glob.glob(joined_files)

# Concatenation of the files
df = pd.concat(map(pd.read_csv, joined_list), ignore_index=True)

# Content of the concatenated files are written into a single newly created file named 'combined.csv'
combined = df
combined.to_csv("combined.csv", index=False)

# File 'combined.csv' is read into a dataFrame
dataFrame = pd.read_csv('combined.csv')
# NaN valued cells are filled with 0 instead
dataFrame = dataFrame.fillna(0)

# To avoid problems that may occur because of data shortage
# only players who played at least 10 minutes per game through a 82 game season are added to the dataframe
# That's why players who had a total of at least 820 minutes played(MP) are selected
dataFrame = dataFrame[dataFrame.MP >= 820]

# To use the primary position of players who can play at multiple positions
# unique values on 'Pos' column are found, based on that all dual values are replaced with only the primary positions

# print('Possible positions a player can play at:' ,dataFrame.Pos.unique())

dataFrame = dataFrame.replace("PG-SG", "PG")
dataFrame = dataFrame.replace("SG-PG", "SG")
dataFrame = dataFrame.replace("SG-SF", "SG")
dataFrame = dataFrame.replace("SF-SG", "SF")
dataFrame = dataFrame.replace("SF-PF", "SF")
dataFrame = dataFrame.replace("PF-SF", "PF")
dataFrame = dataFrame.replace("PF-C", "PF")
dataFrame = dataFrame.replace("C-PF", "C")
dataFrame = dataFrame.replace("SG-PF", "SG")

# All values are rounded up to their decimally double-digit representations
dataFrame = dataFrame.round({'PTS': 2, 'TRB': 2, 'ORB': 2, 'AST': 2, 'STL': 2})
dataFrame = dataFrame.round({'BLK': 2, 'FG': 2, 'FGA': 2, 'FG%': 2, '3P': 2})
dataFrame = dataFrame.round({'3PA': 2, '3P%': 2, '2P': 2, '2PA': 2, '2P%': 2})
dataFrame = dataFrame.round({'FT': 2, 'FTA': 2, 'FT%': 2, 'PF': 2, 'TOV': 2, 'HGT': 2})

# Columns which are not going to be used for prediction are removed from the dataframe
dataFrame = dataFrame.drop(columns='Rk')
dataFrame = dataFrame.drop(columns='Player')
dataFrame = dataFrame.drop(columns='Age')
dataFrame = dataFrame.drop(columns='Tm')
dataFrame = dataFrame.drop(columns='G')
dataFrame = dataFrame.drop(columns='GS')
dataFrame = dataFrame.drop(columns='MP')
dataFrame = dataFrame.drop(columns='DRB')

# print(dataFrame)

# Dispersion based on which position the players in the dataFrame play is examined
#print(dataFrame.loc[:, 'Pos'].value_counts())

# Every feature's average values for each position are shown on a small dataframe named 'summary_df'
summary_df = dataFrame.groupby('Pos').mean()
summary_df = summary_df.round(decimals=3)
#print(summary_df)


# Based on 'summary_df' averages for 5 primary statistics on each position is visualized on a bar chart
def bar_chart():
    bar_chart_df = summary_df[['PTS', 'TRB', 'AST', 'STL', 'BLK']]
    bar_chart_df.plot(kind='bar', figsize = (10, 6), title='Bar Chart of Main Stats across all 5 Positions')
    plt.show()

# Calling the bar_chart function
#bar_chart()

# Height average visualized on a seperate bar chart for each position
def height_bar():
    height_bar_df = summary_df[['HGT']]
    height_bar_df.plot(kind='bar', figsize = (10, 6), title='Bar Chart of Height Averages(m) across all 5 Positions')
    plt.show()

# Calling the height_bar function
#height_bar()

# Dispersion patterns of 5 primary statistics and height data on each other accorging to positions visulaized on a seaborn pair plot
def seaborn():
    sns_df = dataFrame[['PTS', 'TRB', 'AST', 'STL', 'BLK', 'Pos', 'HGT']]
    sns_df = sns_df.reset_index()
    sns_df = sns_df.drop('index', axis=1)
    sns_plot = sns.pairplot(sns_df, hue='Pos', size=2)

    plt.show()

# Calling the seaborn function
#seaborn()


# 'Pos' column is dropped from x-axis and positioned on the y-axis to be shown across other rows
X = dataFrame.drop('Pos', axis=1)
y = dataFrame.loc[:, 'Pos']

# Position names valued in a dictionary as their classic numerical values for the confusion matrix
position_dictionary = {"PG": 1,"SG": 2,"SF": 3,"PF": 4,"C": 5}
y = y.map(position_dictionary).values.reshape(-1,1)

# Splitting data into test and training values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)

# Scaling data across all columns to increase learning performance
scaler = StandardScaler()
X_scaler = scaler.fit(X_train)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)


# Making predictions with a Decision Trees model and evaluating the results
def decision_tree():
    from sklearn import tree
    # Creation of the model
    dt1_model = tree.DecisionTreeClassifier(random_state=1)
    dt1_model = dt1_model.fit(X_train_scaled, y_train)
    # Making predictions
    predictions = dt1_model.predict(X_test_scaled)

    # Model's accuracy percentage
    model_accuracy_score = accuracy_score(y_test, predictions)
    model_accuracy_score = model_accuracy_score * 100
    print('Accuracy score for Decision Tree model: %', "{:.2f}".format(model_accuracy_score))

    # Importance of features on learning
    model_importances = pd.DataFrame(dt1_model.feature_importances_,
                                 index = X_train.columns, columns=['Importance of features for Decision Tree model']
                                     ).sort_values('Importance of features for Decision Tree model', ascending=False)
    print(model_importances)

    # Confusion Matrix for Decision Trees model
    plot_confusion_matrix(dt1_model, X_test_scaled, y_test)
    plt.title('Confusion Matrix for Decision Trees model')
    plt.show()

    # Classification report for Decision Trees model
    print("\n            Classification Report for Decision Tree model")
    model_class_report = classification_report(y_test, predictions, target_names = ['PG', 'SG', 'SF', 'PF', 'C'])
    print(model_class_report)

# Calling decision_tree function
# decision_tree()

'''
# ///// Hyperparameter optimization with GridSearchCV module
rf1_grid = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}
CV_rfc = GridSearchCV(estimator=rf1_grid, param_grid=param_grid, cv= 5)
CV_rfc.fit(X_train, y_train)

print(CV_rfc.best_params_)

'''


# Making predictions with a Random Forest model and evaluating the results
def random_forest():
    from sklearn.ensemble import RandomForestClassifier
    # Creation of the model
    rf1_model = RandomForestClassifier(n_estimators=200, random_state=1, max_depth=8,
                                       max_features='auto', criterion ='gini')
    rf1_model = rf1_model.fit(X_train_scaled, y_train)
    # Making predictions
    predictions = rf1_model.predict(X_test_scaled)

    # Model's accuracy percentage
    model_accuracy_score = accuracy_score(y_test, predictions)
    model_accuracy_score = model_accuracy_score * 100
    print('Accuracy score for Random Forest model: %', "{:.2f}".format(model_accuracy_score))

    # Importance of features on learning
    model_importances = pd.DataFrame(rf1_model.feature_importances_,
                                 index = X_train.columns, columns=['Importance of features for Random Forest model']
                                     ).sort_values('Importance of features for Random Forest model', ascending=False)
    print(model_importances)

    # Confusion Matrix for Random Forest model
    plot_confusion_matrix(rf1_model, X_test_scaled, y_test)
    plt.title('Confusion Matrix for Random Forest model')
    plt.show()

    # Classification report for Random Forest model
    print("\n            Classification Report for Random Forest model")
    model_class_report = classification_report(y_test, predictions, target_names = ['PG', 'SG', 'SF', 'PF', 'C'])
    print(model_class_report)

# Calling random_forest function
# random_forest()


# Making predictions with a Support Vector Machines(SVMs) model and evaluating the results

def svm():
    from sklearn import svm
    from sklearn.svm import SVC
    # Creation of the model
    svm1_model = svm.SVC(kernel='linear', random_state=1)
    svm1_model = svm1_model.fit(X_train_scaled, y_train)
    # Making predictions
    predictions = svm1_model.predict(X_test_scaled)

    # Model's accuracy percentage
    model_accuracy_score = accuracy_score(y_test, predictions)
    model_accuracy_score = model_accuracy_score * 100
    print('Accuracy score for Support Vector Machine model: %', "{:.2f}".format(model_accuracy_score))

    # Since there isn't a '.feature_importances_' method for Support Vector Machines, to visualize these values a plot is formed
    
    def f_importances(coef, names, top=-1):
        imp = coef
        imp, names = zip(*sorted(list(zip(imp, names))))
        # Tüm özniteleliklerin gösterilmesi
        if top == -1:
            top = len(names)
        plt.barh(range(top), imp[::-1][0:top], align='center')
        plt.yticks(range(top), names[::-1][0:top])
        plt.title('Feature Importances for Support Vector Machine model')
        plt.show()
    feature_names = ['PTS', 'TRB', 'ORB', 'AST', 'STL', 'BLK', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%',
                     '2P', '2PA', '2P%', 'FT', 'FTA', 'FT%', 'PF', 'TOV', 'HGT']
    # Plot indicating each feature's importance on learning
    f_importances(abs(svm1_model.coef_[0]), feature_names)

    # Confusion Matrix for Support Vector Machines model
    plot_confusion_matrix(svm1_model, X_test_scaled, y_test)
    plt.title('Confusion Matrix for Support Vector Machine model')
    plt.show()

    # Classification report for Support Vector Machines model
    print("\n            Classification Report for Support Vector Machine model")
    model_class_report = classification_report(y_test, predictions, target_names = ['PG', 'SG', 'SF', 'PF', 'C'])
    print(model_class_report)

# Calling svm function
# svm()


from sklearn.ensemble import GradientBoostingClassifier

# learning_rate plays a crucial role in Gradient Boosting Tree models' learning pace
# so to find the optimal value for this parameter, evalutions are made on values between 0 and 1 with 5% intervals
# The most fitting value is used in the model
for learning_rate in range (5,101,5):
    learning_rate = "{:.2f}".format(learning_rate/100)
    learning_rate = float(learning_rate)
    gbt1_model = GradientBoostingClassifier(n_estimators=20, learning_rate=learning_rate, max_features=5, max_depth=3, random_state=0)
    # Fit the model
    gbt1_model.fit(X_train_scaled, y_train.ravel())
    print("Learning rate: ", learning_rate)
    # Score the model
    print("Accuracy score (training): {0:.3f}".format(
        gbt1_model.score(
            X_train_scaled,
            y_train.ravel())))
    print("Accuracy score (validation): {0:.3f}".format(
        gbt1_model.score(
            X_test_scaled,
            y_test.ravel())))


# Making predictions with a Gradient Boosted Trees model and evaluating the results
def gbt():
    # Creation of the model
    gbt1_model = GradientBoostingClassifier(n_estimators=20, learning_rate=0.15, max_depth=3, random_state=1)
    gbt1_model = gbt1_model.fit(X_train_scaled, y_train)
    # Making predictions
    predictions = gbt1_model.predict(X_test_scaled)

    # Model's accuracy percentage
    model_accuracy_score = accuracy_score(y_test, predictions)
    model_accuracy_score = model_accuracy_score * 100
    print('\nAccuracy score for Gradient Boosted Trees model: %', "{:.2f}".format(model_accuracy_score))

    # Importance of features on learning
    model_importances = pd.DataFrame(gbt1_model.feature_importances_,
                                 index = X_train.columns, columns=['Importance of features for Gradient Boost '
                                                                   'Trees model']
                                 ).sort_values('Importance of features for Gradient Boost Trees model', ascending=False)
    print(model_importances)

    # Confusion Matrix for Gradient Boosted Trees model
    plot_confusion_matrix(gbt1_model, X_test_scaled, y_test)
    plt.title('Confusion Matrix for Gradient Boosted Trees model')
    plt.show()

    # Classification report for Gradient Boosted Trees model
    print("\n            Classification Report for Gradient Boosted Trees model")
    model_class_report = classification_report(y_test, predictions, target_names = ['PG', 'SG', 'SF', 'PF', 'C'])
    print(model_class_report)
    
# Calling gbt function
# gbt()
