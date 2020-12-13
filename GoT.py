import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

#read csv data into dataframe
battledf = pd.read_csv('battles.csv')
#function that replaces missing entries with NaN in columns with string data
def fillmissing(df):
    for col in df:
        if(df[col].dtype=='object'):
            df[col].replace('','NaN', inplace=True)
        else:
            continue
    return df
#fill missing entries with NaN in columns with string data
df_filled = fillmissing(battledf)

#drop columns that are irrelevant to the hypotheses being tested
df_dropped = df_filled.drop(['name','year','battle_number','attacker_3','attacker_4','defender_2','defender_3','defender_4','major_death','major_capture','attacker_commander','defender_commander','attacker_size','defender_size','location','note'], axis=1)

#only keep 'attacker_1', 'attacker_outcome', and 'battle_type' columns for first catplot
df_dropped2 = df_dropped.drop(['attacker_2','defender_1','summer','region'], axis=1)
#only consider features(rows) for pitched battles
df_pitched = df_dropped2[df_dropped2.battle_type == 'pitched battle']
df_pitched.rename(columns={'attacker_1':'Attacking House','attacker_outcome':'Outcome'}, inplace=True)
#catplot of number of battles won and lost by attacking houses
plot1 = sns.catplot(data=df_pitched, y='Attacking House', col='Outcome', kind="count")
plt.subplots_adjust(top=0.9)
plot1.fig.suptitle('Pitched Battles won and lost by Attacking Houses')
plt.show(plot1)
plot1.savefig("Attack_Outcome.png")

#only keep 'defender_1', 'attacker_outcome', and 'battle_type' columns for second catplot
df_dropped3 = df_dropped.drop(['attacker_1','attacker_2','summer','region'], axis=1)
df_pitched2 = df_dropped3[df_dropped3.battle_type == 'pitched battle']
df_pitched2.rename(columns={'defender_1':'Defending House','attacker_outcome':'Outcome'}, inplace=True)
#convert attacker outcome column to defender outcome column by changing 'win' to 'loss' and vice versa
for i,row in df_pitched2.iterrows():
    if df_pitched2.at[i, 'Outcome'] == 'win':
        df_pitched2.at[i, 'Outcome'] = 'loss'
    else:
        df_pitched2.at[i, 'Outcome'] = 'win'
#catplot of number of battles won and lost by defending houses
plot2 = sns.catplot(data=df_pitched2, y='Defending House', col='Outcome', kind="count")
plt.subplots_adjust(top=0.9)
plot2.fig.suptitle('Pitched Battles won and lost by Defending Houses')
plt.show(plot2)
plot2.savefig("Defend_Outcome.png")

#keymap for converting names of kings to numerical entries for second test
king_keymap = {"attacker_king": {"Joffrey/Tommen Baratheon": 1, "Robb Stark": 2, "Balon/Euron Greyjoy": 3, "Stannis Baratheon": 4},
                "defender_king": {"Joffrey/Tommen Baratheon": 1, "Robb Stark": 2, "Balon/Euron Greyjoy": 3, "Stannis Baratheon": 4,
                                  "Renly Baratheon": 5, "Mance Rayder": 6}}
#keymap for converting names of houses to numerical entries for second test
house_keymap = {"attacker_1": {"Stark":1, "Lannister":2, "Greyjoy":3, "Baratheon":4, "Frey":5, "Bolton":6,
                               "Brotherhood without Banners":7, "Brave Companions":8, "Bracken":9, "Free folk":10, "Darry":11},
                "attacker_2": {"Tully":12, "Karstark":13, "Frey":5, "Lannister":2, "Bolton":6, "Thenns":14, "Greyjoy":3},
                "defender_1": {"Stark":1, "Lannister":2, "Greyjoy":3, "Baratheon":4, "Tully":12, "Bolton":6, "Blackwood": 15,
                               "Tyrell":16, "Brave Companions":8, "Night's Watch":17, "Darry":11, "Mallister":18}}
#replace the columns with names of kings and houses with unique numbers using the keymaps
df_dropped.replace(king_keymap, inplace=True)
df_dropped.replace(house_keymap, inplace=True)

#drop the last row because both 'attacker_outcome' and ''battle_type' have missing values in this row
df_dropped.drop(df_dropped.tail(1).index, inplace=True)
#Convert the string entries in 'attacker_outcome', 'battle_type', and 'region' column to numbers for second test
le = LabelEncoder()
df_dropped['attacker_outcome'] = le.fit_transform(df_dropped['attacker_outcome'])
df_dropped['battle_type'] = le.fit_transform(df_dropped['battle_type'])
df_dropped['region'] = le.fit_transform(df_dropped['region'])
#finally, replace all NaN values in munged dataframe with 1.5
df_final = df_dropped.fillna(1.5)
#Plot heatmap to correlate selected features for second test
Corr = df_final[['attacker_king','defender_king','attacker_outcome','attacker_1','defender_1','battle_type','region']]
sns.set()
CorrelationMap = sns.heatmap(Corr.corr(),annot = True)
CorrelationMap.set(title = 'Correlation between attacker outcome and selected features')
plt.show()
FigMap = CorrelationMap.get_figure()
FigMap.savefig("HeatMapCorr.png")

#function for performing logistic regression algorithm on data and printing results
def log_regression(data, target):
    # Split train and test data Fit train data to logistic regression model Run predictions on test data
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.4, stratify=target)
    LogReg = LogisticRegression()
    LogReg.fit(x_train, y_train)
    y_pred = LogReg.predict(x_test)

    # Print accuracy score and classification report
    print('The accuracy score is:', accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    # Output graph of confusion matrix
    cm = pd.DataFrame(confusion_matrix(y_test, y_pred), columns=["Loss", "Win"], index=[["Loss", "Win"]])
    heat_cm = sns.heatmap(cm, annot=True)
    plt.xlabel("Prediction")
    plt.ylabel("Truth")
    plt.title('Accuracy Score: %.6f' % accuracy_score(y_test, y_pred))
    plt.show(heat_cm)

#drop 'battle_type' and 'attacker_outcome' from data to make predictions based on region
data1 = df_final.drop(['attacker_outcome','battle_type'], axis=1)
target1 = df_final['attacker_outcome']
print('Predicting attacker outcome based on region and selected features:')
log_regression(data1, target1)

#drop 'region' and 'attacker_outcome' from data to make predictions based on battle type
data2 = df_final.drop(['attacker_outcome','region'], axis=1)
target2 = df_final['attacker_outcome']
print('Predicting attacker outcome based on battle type and selected features:')
log_regression(data2, target2)