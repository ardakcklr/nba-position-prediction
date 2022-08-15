# importing libraries
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



# Oluşabilecek uyarıların göz ardı edilmesi
import warnings
warnings.filterwarnings('ignore')

# 'nba' ile başlayan tüm csv dosyalarının alınması
joined_files = os.path.join(r"C:\Users\rdkck\PycharmProjects\CIDP_Project2", "nba*.csv")

# Bu dosyaların bir listesinin döndürülemsi
joined_list = glob.glob(joined_files)

# Dosyaların birleştirilmesi
df = pd.concat(map(pd.read_csv, joined_list), ignore_index=True)

# Birleştirilen dosyaların yeni oluşturulan 'combined.csv' adlı dosyaya yazdırılması
combined = df
combined.to_csv("combined.csv", index=False)

# combined.csv dosyasının bir dataFrame'e okunması
dataFrame = pd.read_csv('combined.csv')
# NaN değerlerin yerlerinin 0 ile doldurulması
dataFrame = dataFrame.fillna(0)

# Veri yetersizliği nedeniyle oluşabilecek sorunları önlemek amacıyla
# bir maçta ortalama olarak en az 10 dakika süre alan ve bunu sezon geneline yayabilen oyuncular dataFrame'e eklenmiştir.
# Bir sezon 82 maç üzerinden düşünülerek oynanan toplam dakikanın 820'den büyük olması istenmiştir.
dataFrame = dataFrame[dataFrame.MP >= 820]

# Birden fazla pozisyonda oynayan oyuncuların birincil pozisyonlarını almak için öncelikle
# 'Pos' sütunu üzerindeki unique değerler bulunmuş ve ikili değerler birincil pozisyonlar ile değiştirilmiştir

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

# Bütün değerler ikili ondalıklarına yuvarlanacak şekilde değiştirilmiştir
dataFrame = dataFrame.round({'PTS': 2, 'TRB': 2, 'ORB': 2, 'AST': 2, 'STL': 2})
dataFrame = dataFrame.round({'BLK': 2, 'FG': 2, 'FGA': 2, 'FG%': 2, '3P': 2})
dataFrame = dataFrame.round({'3PA': 2, '3P%': 2, '2P': 2, '2PA': 2, '2P%': 2})
dataFrame = dataFrame.round({'FT': 2, 'FTA': 2, 'FT%': 2, 'PF': 2, 'TOV': 2, 'HGT': 2})

# Tahmin için gerekli olmayacak sütunlar dataFrame'den silinmiştir
dataFrame = dataFrame.drop(columns='Rk')
dataFrame = dataFrame.drop(columns='Player')
dataFrame = dataFrame.drop(columns='Age')
dataFrame = dataFrame.drop(columns='Tm')
dataFrame = dataFrame.drop(columns='G')
dataFrame = dataFrame.drop(columns='GS')
dataFrame = dataFrame.drop(columns='MP')
dataFrame = dataFrame.drop(columns='DRB')

# print(dataFrame)

# dataFrame'deki oyuncuların pozisyonlarına göre dağılımları incelenmiştir
#print(dataFrame.loc[:, 'Pos'].value_counts())

# Her bir poziyon için çıkarılan özniteliklerin ortalama değerlerini gösteren
# özet şeklinde bir dataFrame oluşturulmuştur
summary_df = dataFrame.groupby('Pos').mean()
summary_df = summary_df.round(decimals=3)
#print(summary_df)


# Bu ortalama dataFrame'i üzerinden belirli verilerin verisetimizde nasıl dağıldığı bar chart
# ve seaborn pair plot kullanılarak görselleştirilmiştir

# 5 ana istatistiğin ortlama olarak pozisyonlar üzerinde bar chart ile gösterilmesi
def bar_chart():
    bar_chart_df = summary_df[['PTS', 'TRB', 'AST', 'STL', 'BLK']]
    bar_chart_df.plot(kind='bar', figsize = (10, 6), title='Bar Chart of Main Stats across all 5 Positions')
    plt.show()

#bar_chart()

# Ortalama boy ölçülerin pozisyonlar üzerinde bar chart ile gösterilmesi
def height_bar():
    height_bar_df = summary_df[['HGT']]
    height_bar_df.plot(kind='bar', figsize = (10, 6), title='Bar Chart of Height Averages(m) across all 5 Positions')
    plt.show()

#height_bar()

# 5 ana istatistiğin ve boy değerlerinin birbirileri üzerinde ve pozisyonlara göre nasıl
# dağılım gösterdiğinin seaborn pair plot ile görselleştirilmesi
def seaborn():
    sns_df = dataFrame[['PTS', 'TRB', 'AST', 'STL', 'BLK', 'Pos', 'HGT']]
    sns_df = sns_df.reset_index()
    sns_df = sns_df.drop('index', axis=1)
    sns_plot = sns.pairplot(sns_df, hue='Pos', size=2)

    plt.show()

#seaborn()


# Pozisyon sütununun x ekseninden düşürülerek diğer satırlar üzerinde
# gösterilecek şekilde y eksenine alınması
X = dataFrame.drop('Pos', axis=1)
y = dataFrame.loc[:, 'Pos']

# Karmaşıklık matrisi için pozisyon isimlerinin klasikleşmiş nümerik gösterimlerine çevrilmesi
position_dictionary = {"PG": 1,"SG": 2,"SF": 3,"PF": 4,"C": 5}
y = y.map(position_dictionary).values.reshape(-1,1)

# Verilerin test ve eğitim verisetleri olarak ayrıştırılması
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)

# Bütün sütunlar üzerindeki değerlerin öğrenmeyi artırmak amacıyla benzer şekilde ölçeklendirilmesi
scaler = StandardScaler()
X_scaler = scaler.fit(X_train)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)


# Decision Trees ile oluşturulan model ile tahminleme yapılması ve modelin değerlendirilmesi
def decision_tree():
    from sklearn import tree
    # Modelin yaratılması
    dt1_model = tree.DecisionTreeClassifier(random_state=1)
    dt1_model = dt1_model.fit(X_train_scaled, y_train)
    # Tahminlemeler yapılması
    predictions = dt1_model.predict(X_test_scaled)

    # Modelin yüzdelik olarak doğruluk değeri
    model_accuracy_score = accuracy_score(y_test, predictions)
    model_accuracy_score = model_accuracy_score * 100
    print('Accuracy score for Decision Tree model: %', "{:.2f}".format(model_accuracy_score))

    # Özniteliklerin modelin öğrenmesi üzerindeki önemleri
    model_importances = pd.DataFrame(dt1_model.feature_importances_,
                                 index = X_train.columns, columns=['Importance of features for Decision Tree model']
                                     ).sort_values('Importance of features for Decision Tree model', ascending=False)
    print(model_importances)

    # Decision Tree modeli için karmaşıklık matrisi
    plot_confusion_matrix(dt1_model, X_test_scaled, y_test)
    plt.title('Confusion Matrix for Decision Trees model')
    plt.show()

    # Decision Tree modeli için sınıflandırma raporu
    print("\n            Classification Report for Decision Tree model")
    model_class_report = classification_report(y_test, predictions, target_names = ['PG', 'SG', 'SF', 'PF', 'C'])
    print(model_class_report)

# decision_tree()

'''
# ///// GridSearchCV modulü ile hyperparametre optimizasyonu
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


# Random Forests ile oluşturulan model ile tahminleme yapılması ve modelin değerlendirilmesi
def random_forest():
    from sklearn.ensemble import RandomForestClassifier
    # Modelin yaratılması
    rf1_model = RandomForestClassifier(n_estimators=200, random_state=1, max_depth=8,
                                       max_features='auto', criterion ='gini')
    rf1_model = rf1_model.fit(X_train_scaled, y_train)
    # Tahminlemeler yapılması
    predictions = rf1_model.predict(X_test_scaled)

    # Modelin yüzdelik olarak doğruluk değeri
    model_accuracy_score = accuracy_score(y_test, predictions)
    model_accuracy_score = model_accuracy_score * 100
    print('Accuracy score for Random Forest model: %', "{:.2f}".format(model_accuracy_score))

    # Özniteliklerin modelin öğrenmesi üzerindeki önemleri
    model_importances = pd.DataFrame(rf1_model.feature_importances_,
                                 index = X_train.columns, columns=['Importance of features for Random Forest model']
                                     ).sort_values('Importance of features for Random Forest model', ascending=False)
    print(model_importances)

    # Random Forest modeli için karmaşıklık matrisi
    plot_confusion_matrix(rf1_model, X_test_scaled, y_test)
    plt.title('Confusion Matrix for Random Forest model')
    plt.show()

    # Random Forest modeli için sınıflandırma raporu
    print("\n            Classification Report for Random Forest model")
    model_class_report = classification_report(y_test, predictions, target_names = ['PG', 'SG', 'SF', 'PF', 'C'])
    print(model_class_report)

# random_forest()


# Support Vector Machines (SVMs) ile oluşturulan model ile tahminleme yapılması ve modelin değerlendirilmesi

def svm():
    from sklearn import svm
    from sklearn.svm import SVC
    # Modelin yaratılması
    svm1_model = svm.SVC(kernel='linear', random_state=1)
    svm1_model = svm1_model.fit(X_train_scaled, y_train)
    # Tahminlemeler yapılması
    predictions = svm1_model.predict(X_test_scaled)

    # Modelin yüzdelik olarak doğruluk değeri
    model_accuracy_score = accuracy_score(y_test, predictions)
    model_accuracy_score = model_accuracy_score * 100
    print('Accuracy score for Support Vector Machine model: %', "{:.2f}".format(model_accuracy_score))

    # Support Vector Machineler aynı '.feature_importances_' fonksiyonu bulunmamaktadır bu nedenle
    # bu değerleri görselleştirmek için bir plot oluşturulacaktır
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
    # Özniteliklerin modelin öğrenmesi üzerindeki önemleri işaret eden plot grafiği
    f_importances(abs(svm1_model.coef_[0]), feature_names)

    # Support Vector Machine modeli için karmaşıklık matrisi
    plot_confusion_matrix(svm1_model, X_test_scaled, y_test)
    plt.title('Confusion Matrix for Support Vector Machine model')
    plt.show()

    # Support Vector Machine modeli için sınıflandırma raporu
    print("\n            Classification Report for Support Vector Machine model")
    model_class_report = classification_report(y_test, predictions, target_names = ['PG', 'SG', 'SF', 'PF', 'C'])
    print(model_class_report)

# svm()


from sklearn.ensemble import GradientBoostingClassifier

# Gradient Boosting Tree modellerin öğrenme hızında kilit rol oynayan learning_rate değerinin en optimal
# değerini bulmak amacıyla 0 ve 1 arasında %5'lik aralıklarla değerlendirmeler yapılmıştır
# En uygun görülen değer modelde kullnılmıştır
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


# Gradient Boosted Trees (GBTs) ile oluşturulan model ile tahminleme yapılması ve modelin değerlendirilmesi
def gbt():
    # Modelin yaratılması
    gbt1_model = GradientBoostingClassifier(n_estimators=20, learning_rate=0.15, max_depth=3, random_state=1)
    gbt1_model = gbt1_model.fit(X_train_scaled, y_train)
    # Tahminlemeler yapılması
    predictions = gbt1_model.predict(X_test_scaled)

    # Modelin yüzdelik olarak doğruluk değeri
    model_accuracy_score = accuracy_score(y_test, predictions)
    model_accuracy_score = model_accuracy_score * 100
    print('\nAccuracy score for Gradient Boosted Trees model: %', "{:.2f}".format(model_accuracy_score))

    # Özniteliklerin modelin öğrenmesi üzerindeki önemleri
    model_importances = pd.DataFrame(gbt1_model.feature_importances_,
                                 index = X_train.columns, columns=['Importance of features for Gradient Boost '
                                                                   'Trees model']
                                 ).sort_values('Importance of features for Gradient Boost Trees model', ascending=False)
    print(model_importances)

    # Gradient Boosted Trees modeli için karmaşıklık matrisi
    plot_confusion_matrix(gbt1_model, X_test_scaled, y_test)
    plt.title('Confusion Matrix for Gradient Boosted Trees model')
    plt.show()

    # Gradient Boosted Trees modeli için sınıflandırma raporu
    print("\n            Classification Report for Gradient Boosted Trees model")
    model_class_report = classification_report(y_test, predictions, target_names = ['PG', 'SG', 'SF', 'PF', 'C'])
    print(model_class_report)

# gbt()
