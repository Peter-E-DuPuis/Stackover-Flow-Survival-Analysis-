import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter

#kmf object for use throughout
kmf = KaplanMeierFitter()

#### Cleaning Procedure-------------------------

#Lodaing csv from local machine (see repository for data, originally from Kaggle)
ans = pd.read_csv("~\\Answers.csv",encoding='latin-1')
qus = pd.read_csv("~\Questions.csv",encoding='latin-1')

#Reducing answers to best score for each question
ans = ans.sort_values(['ParentId','Score'],ascending=[True,False])
ans = ans.drop_duplicates(subset='ParentId')

#Merging questions and answers together using left outer-join
sf = pd.merge(qus,ans,how = 'left', left_on='Id', right_on='ParentId')

# Altering the answer scores for modeling. For our purposes, we will have any question with 3 or few votes be considered "unanswered"
sf['event'] = sf.Score_y >= 3

#Creating time variables, which will show # of hours it takes for a question to receive its highest score.
sf['ans_date'] = pd.to_datetime(sf['CreationDate_y'])
sf['ask_date'] = pd.to_datetime(sf['CreationDate_x'])
sf['duration'] = sf['ans_date'] - sf['ask_date'] 
sf['duration_min'] = sf['duration'].dt.total_seconds()/60
sf['duration_hr'] = sf["duration_min"] / 60
sf['duration_day'] = sf['duration_hr'] / 24
sf['duration_hr'] = sf.duration_hr.round(2)
sf['duration_day'] = sf.duration_day.round(3)
sf = sf[sf['duration_hr'] >= 0] # removing negative values.
sf= sf[sf['duration_day'] >= 0]

#Replacing Nan with 0 and assuming they are unanswered
sf["Score_y"] = sf.Score_y.fillna(0)

#Times with Nan will be replaced with 5999.99, which is slightly higher than highest (250 for days) 
sf['duration_hr'] = sf['duration_hr'].fillna(5999.99)
sf['duration_day']= sf['duration_day'].fillna(250)

 #### Summary Statistics --------------------------------------------

#Distribution of hours
sns.set_style('whitegrid')
sns.kdeplot(sf['duration_hr'], bw=.5)

#Distribution of question length
sns.set_style('whitegrid')
sns.kdeplot(sf['q_length'], bw=.5)

#Covarience Heatmap
cm = sf.corr()
sns.heatmap(cm,
            xticklabels=cm.columns,
            yticklabels=cm.columns)


 #### Survival Anaylsis --------------------------------------------------------------------------------

#Using "Kaplan Meir" method on cleaned data
T = sf['duration_day'].values.astype('int64')
E = sf['event'].values
kmf.fit(T,E)

#plotting
kmf.survival_function_.plot()
plt.title('Survival Function of Python Questions on Stackover Flow')
plt.ylabel("est. probability of survival (qustion NOT being answered) $\hat{S}(t)$")
plt.xlabel("time (days) $t$")
#plt.show()

kmf.median_ # median number of days to have questioned answered (i.e. 50% chance your question is answered within 148 days)

# Survival Fuction by question lenght(binned at short (<= 25%), mideum (26%-50%), mid-long(51%-75%), long (>75%))
sf['q_length'] = sf['Body_x'].str.len()
sf['an_length'] = sf['Body_y'].str.len()
sf['t_length'] = sf['Title'].str.len()
sf['q_len'] = pd.qcut(sf.q_length, 2, labels=["short","long"]) 

#Survival by question length
ax = plt.subplot(111)

short = (sf["q_len"] == "short")
kmf.fit(T[short], event_observed=E[short], label="Shorter Questions")
kmf.plot(ax=ax, ci_force_lines=True)
kmf.fit(T[~short], event_observed=E[~short], label="Longer Questions")
kmf.plot(ax=ax, ci_force_lines=True)

plt.ylim(0, 1);
plt.title("Lifespans of different Question Lenghts")

#Median survival time by question length
ax = plt.subplot(111)

t = np.linspace(0, 500, 501)
kmf.fit(T[short], event_observed=E[short], timeline=t, label="Shorter Questions")
ax = kmf.plot(ax=ax)
print("Median survival time of short questions:", kmf.median_)

kmf.fit(T[~short], event_observed=E[~short], timeline=t, label="Longer Questions")
ax = kmf.plot(ax=ax)
print("Median survival time of Longer Questions:", kmf.median_)

plt.ylim(0,1)
plt.title("Lifespans of different Question types in First 500 Days")

# Test of significances between Question Types
from lifelines.statistics import logrank_test

results = logrank_test(T[short], T[~short], E[short], E[~short], alpha=.99)

results.print_summary()

# Applying output to a hazord curve. 
from lifelines import NelsonAalenFitter
naf = NelsonAalenFitter()

naf.fit(T,event_observed=E)
naf.plot()

#By question length
naf.fit(T[short], event_observed=E[short], label="Shorter Questions")
ax = naf.plot(loc=slice(0, 200))
naf.fit(T[~short], event_observed=E[~short], label="Longer Questions")
naf.plot(ax=ax, loc=slice(0, 200))
plt.title("Cumulative hazard function by Question Length (up to 2000= days)")


# Aalen's Additive Model
from lifelines import CoxPHFitter
cph= CoxPHFitter()

#Covariance matrix
import patsy
sfm = patsy.dmatrix('Score_x + t_length + q_length +an_length-1', sf, return_type='dataframe')

#tidying date
sfm['T'] = sf['duration_day']
sfm['E'] = sf['event']
reduce = int(len(sfm.index)/15)
sfm = sfm.sample(reduce)
del(E,T,ans,cm,qus,sf,short,t,reduce)

#chp fitting
cph.fit(sfm,duration_col='T',event_col='E')
cph.print_summary()
cph.plot()

#cross validation
test = k_fold_cross_validation(cph, sfm, 'T', event_col='E', k=3)
print(test)
print(np.mean(test))
print(np.std(test))


#Note: some code for kmf adpated from tutoral: http://lifelines.readthedocs.io/en/latest/Survival%20analysis%20with%20lifelines.html
