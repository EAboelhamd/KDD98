#!/usr/bin/env python

import pandas as pd
import numpy as np
import sys
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import cross_val_score
import pickle


def learn_vals(df,label,features):
  #Machine learning to fill nominal values in:
  '''
  learn_vals(df,label,features)
  where:
  df = data frame
  label = column to learn
  features = columns to learn from
  '''
  
  #Check that all columns are in data frame
  for item in features:
    if item not in list(df.columns):
      features.remove(item)
      
  if label not in list(df.columns) or sum(df[label].isnull())==0:
    return df
  
  print 'Filling in' , label , 'values...'
  
  et = ExtraTreesClassifier(n_estimators=100, max_depth=None, min_samples_split=1, random_state=0,verbose=True,n_jobs=-1)
 
  labels_train = df[label][df[label].isnull() == 0].values

  features_train = df[features][df[label].isnull() == 0].values
  features_test  = df[features][df[label].isnull()].values

  et.fit(features_train,labels_train)
  labels_test = pd.Series(et.predict(features_test),index = df.index[df[label].isnull()])
  NewCol = pd.concat([ df[label][df[label].isnull()==0] , labels_test] )
  
  df = df.combine_first( pd.DataFrame({label:NewCol}) )

  return df



def ConvertToInteger(string):
  '''
  Converts a string of mixed letters and numbers into an integer
  '''
  IntegerList = [str(x) for x in range(10)]
  output = ''

  if string == ' ':
    return string
  else:
    for n in range(len(string)):
      if string[n] not in IntegerList:
        output+=str(ord(string[n]))
      else:
        output+=string[n]
  
    return int(output)
  


def main():
  '''
  Correct syntax:
  ./CleanData.py inputfile.txt outputfile.p
  
  Where inputfile.txt contains the data to be cleaned, and outputfile.p is where to save the cleaned data.
  '''
  
  
  if len(sys.argv) > 3:
    print 'Usage: ./CleanData.py inputfile.txt outputfile.p'
    sys.exit(0)
  
  # I get 'CParserError: Error tokenizing data; when running pd.read_csv
  # Possibly because of way the file is formatted. Get around this by ignoring lines with error (only around 0.2%)
  df = pd.read_csv(sys.argv[1],error_bad_lines=False,warn_bad_lines=False)
  
  
  '''
  Data cleaning / reformatting
  '''
  
  #List of categorical fields
  Nom_list = [x for x in df.columns if df[x].dtype == np.dtype('O')]
      
  #Reformat columns according to the dictionary at http://kdd.ics.uci.edu/databases/kddcup98/epsilon_mirror/cup98dic.txt
  print 'Reformatting columns...'
  df[['RECINHSE','RECP3','RECPGVG','RECSWEEP','MAJOR','PEPSTRFL','LIFESRC']] = df[['RECINHSE', 'RECP3','RECPGVG','RECSWEEP','MAJOR','PEPSTRFL','LIFESRC']].replace(r'\s+',0,regex=True)
  df[['NOEXCH','RECINHSE', 'RECP3','RECPGVG','RECSWEEP','MAJOR','PEPSTRFL']] = df[['NOEXCH','RECINHSE', 'RECP3','RECPGVG','RECSWEEP','MAJOR','PEPSTRFL']].replace('X',1,regex=True)
  df[['MAILCODE','SOLP3','SOLIH']] = df[['MAILCODE','SOLP3','SOLIH']].replace(r'\s+',1,regex=True)
  df[['SOLP3','SOLIH']] = df[['SOLP3','SOLIH']].apply(lambda x : x > 0)
  df['MAILCODE'].replace('B',0,inplace=True)
  df['ZIP'] = df['ZIP'].apply(lambda x: x.rstrip('-'))
  df['STATE'] = df['STATE'].apply(ConvertToInteger)
  df['MDMAUD'] = df['MDMAUD'].apply(ConvertToInteger)
  
  for n in range(2,25):
    s = 'RFA_'+str(n)
    if s in df.columns:
      df[s] = df[s].apply(ConvertToInteger)
    
  df['DOMAIN'] = df['DOMAIN'].apply(ConvertToInteger)
  df['GENDER'].replace('M',1,inplace=True)
  df['GENDER'].replace('F',0,inplace=True)
  df['GENDER'].replace(r'[AUJC\s]+',np.nan,regex=True,inplace=True)

  df.replace(r'\s+',np.nan, regex=True,inplace=True)

  #Group uncommon titles (<0.1% of occurences)
  #211  = Other
  s = df['TCODE'].value_counts()
  for item in s.keys():
     if s[item] < 0.001*df.shape[0]:
       df['TCODE'].replace(item,'211',inplace=True)
  df['TCODE'] = df['TCODE'].astype(int)

  #Replace missing GENDER values if TCODE gives gender
  for row in df.index[df['GENDER'].isnull()].tolist():
    a = df.loc[row,'TCODE']
    if a == 0 or a == 1:
      df.loc[row,'GENDER'] = 1
    elif a == 2 or a == 3 or a == 28:
      df.loc[row,'GENDER'] = 0
      
  # # #Drop bad address records since only account for 1% of data and address info may be useful.
  MissingAddresses = df.index[df['MAILCODE'] == 0].tolist()
  df = df.drop(MissingAddresses)
  df = df.reindex()

  #Drop irrelevant, duplicated or underfilled records
  print 'Dropping sparse or unneccesary columns...'
  
  Drop_list = ['NOEXCH','OSOURCE','AGEFLAG','AGE','DATASRCE','CHILD03','CHILD07','CHILD12','CHILD18','COLLECT1','VETERANS','BIBLE',
  'CATLG','HOMEE','PETS','CDPLAY','STEREO','PCOWNERS','PHOTO','CRAFTS','FISHER','GARDENIN','BOATS','WALKER','KIDSTUFF','CARDS','PLATES',
  'LIFESRC','MAILCODE','HOMEOWNR','GEOCODE', 'CLUSTER2', 'GEOCODE2', 'RFA_2R', 'RFA_2F', 'RFA_2A', 'MDMAUD_R', 'MDMAUD_F', 'MDMAUD_A']
  
  l1 = ['RDATE_'+str(n) for n in range(3,25)]
  l2 = ['ADATE_'+str(n) for n in range(2,25)]
  # l3 = ['RFA_'+str(n) for n in range(3,25)]
  
  Drop_list.extend(l1)
  Drop_list.extend(l2)
  # Drop_list.extend(l3)
  
  #Drop the columns with more than half the data missing
  
  n=0
  Half_filled = [x for x in df.columns if sum(df[x].isnull()) > 0.5*df.shape[0] ]
  Drop_list.extend(Half_filled)
  
  Drop_list = list(set(Drop_list)) #To remove duplicates, convert to a set
  df.drop(Drop_list,axis = 1,inplace = True)
  print 'Number of dropped Columns = ' , len(Drop_list)


  '''
  Now fill missing data... Use averages for numerical values and machine learning for nominal values.
  '''

  #Fill average data in numerical column
  print 'Filling in using averages...'
  NumericalData = [item for item in df.columns if item not in Nom_list]
  for col in NumericalData:
    df[col] = df[col].astype(float)
    Mean = np.mean(df[col])
    df[col] = df[col].fillna(Mean)
    df[col] = df[col].astype(int)
  
  #Machine learning to fill categorical values in:
  print 'Filling in nominal values...'  
  Labels_list = []
  for item in Nom_list:
    if item in df.columns:
      if sum(df[item].isnull()) > 0:
        Labels_list.append(item)
  
  for item in Labels_list:
    feature_list = [x for x in df.columns if x not in Labels_list]
    df = learn_vals(df,item,feature_list)
    
  for col in df.columns:
    if sum(df[col].isnull()) > 0:
      print 'NaN values remaining in column',col
  
  #Check for correlation between columns, and remove duplicate information
  print 'Dropping highly correlated columns...'
  l = []
  Corr_Mat = df.corr()
  cols = list(Corr_Mat.columns)
  for n in range(Corr_Mat.shape[0]):
    for m in range(n+1,Corr_Mat.shape[0]):
          if abs(Corr_Mat.loc[cols[n],cols[m]]) > 0.9:
              l.append(cols[m])
              
  l = list(set(l))
  if 'TARGET_B' in l:
    l.remove('TARGET_B')
  elif 'TARGET_D' in l:
    l.remove('TARGET_D')
  
  df.drop(l,axis = 1,inplace = True)
  print 'Dropped ',len(l),' columns'
  
  #Set index to df.CONTROLN
  df = df.set_index('CONTROLN')
  out_file = sys.argv[2]
  #Save cleaned data
  pickle.dump(df, open( out_file , "wb" ) )


if __name__ == '__main__':
  main()
