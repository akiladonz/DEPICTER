#!/home/pyenv/versions/py3.7/bin/python


# In[1]:


import numpy as np
import pandas as pd
import csv
import sys
import pickle
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
inFile1=sys.argv[1]
inFile2=sys.argv[2]
inFile3=sys.argv[3]





# In[ ]:


Lines = []
f = open(inFile1,'r')
for line in f:
    data_line = line.rstrip().split('\t')
    Lines.append(data_line)

IUPL_Score =Lines[0]
IUPL = IUPL_Score[0].rstrip().split(',')
New_IUPL_Score = list(map(float, IUPL))


# In[ ]:


Lines = []
f = open(inFile2,'r')
for line in f:
    data_line = line.rstrip().split('\t')
    Lines.append(data_line)

IUPS_Score =Lines[0]
IUPS = IUPS_Score[0].rstrip().split(',')
New_IUPS_Score = list(map(float, IUPS))


# In[ ]:


Lines = []
f = open(inFile3, 'r')
for line in f:
    data_line = line.rstrip().split('\t')
    Lines.append(data_line)

Spot_Score =Lines[0]
spot = Spot_Score[0].rstrip().split(',')
New_Spot_Score = list(map(float, spot))

d=len(New_Spot_Score)

# In[20]:


from sklearn.preprocessing import MinMaxScaler
scaler1 = MinMaxScaler(feature_range=(0, 0.5))
scaler2 = MinMaxScaler(feature_range=(0.5, 1))


# In[21]:


score_prediction= New_Spot_Score
Scaled_Spot_score =[]
score_prediction = np.asarray(score_prediction)
score_prediction = score_prediction.reshape(-1, 1)
scaler1.fit(score_prediction)
scaler2.fit(score_prediction)
b=0
for b in range(0, len(score_prediction), 1):

    if score_prediction[b] < 0.132: sca = scaler1.transform(score_prediction[b].reshape(1, -1))
    elif score_prediction[b] >= 0.132 : sca = scaler2.transform(score_prediction[b].reshape(1, -1))    
    Scaled_Spot_score.append(sca)
    
Scaled_Spot_score = np.concatenate(np.concatenate(Scaled_Spot_score))


# In[22]:


score_prediction= New_IUPL_Score
Scaled_IUPL_score =[]
score_prediction = np.asarray(score_prediction)
score_prediction = score_prediction.reshape(-1, 1)
scaler1.fit(score_prediction)
scaler2.fit(score_prediction)
b=0
for b in range(0, len(score_prediction), 1):

    if score_prediction[b] < 0.4542: sca = scaler1.transform(score_prediction[b].reshape(1, -1))
    elif score_prediction[b] >= 0.4542 : sca = scaler2.transform(score_prediction[b].reshape(1, -1))    
    Scaled_IUPL_score.append(sca)
    
Scaled_IUPL_score = np.concatenate(np.concatenate(Scaled_IUPL_score))


# In[23]:


score_prediction= New_IUPS_Score
Scaled_IUPS_score =[]
score_prediction = np.asarray(score_prediction)
score_prediction = score_prediction.reshape(-1, 1)
scaler1.fit(score_prediction)
scaler2.fit(score_prediction)
b=0
for b in range(0, len(score_prediction), 1):

    if score_prediction[b] < 0.446: sca = scaler1.transform(score_prediction[b].reshape(1, -1))
    elif score_prediction[b] >= 0.446 : sca = scaler2.transform(score_prediction[b].reshape(1, -1))    
    Scaled_IUPS_score.append(sca)
    
Scaled_IUPS_score = np.concatenate(np.concatenate(Scaled_IUPS_score))


# In[24]:


randompred = New_IUPL_Score
IUPL_Binary= [] 
binary=0
b = 0
for b in range(0, len(randompred), 1):
    if randompred[b] >=0.4542: binary= 1 
    else: binary=0
    IUPL_Binary.append(binary)
sum(IUPL_Binary)


# In[25]:


randompred = New_IUPS_Score
IUPS_Binary= [] 
binary=0
b = 0
for b in range(0, len(randompred), 1):
    if randompred[b] >= 0.446: binary= 1 
    else: binary=0
    IUPS_Binary.append(binary)
sum(IUPS_Binary)


# In[26]:


randompred = New_Spot_Score
Spot_Binary= [] 
binary=0
b = 0
for b in range(0, len(randompred), 1):
    if randompred[b] >= 0.132: binary= 1 
    else: binary=0
    Spot_Binary.append(binary)
sum(Spot_Binary)


# In[27]:


Spot_PredictionScore=Scaled_Spot_score
Spot_BinaryPredictions=Spot_Binary
IUPL_PredictionScore=Scaled_IUPL_score
IUPL_BinaryPredictions=IUPL_Binary


# In[28]:


Spot_PredictionScore_byProteins= Spot_PredictionScore
Spot_BinaryPrediction_byProteins= Spot_BinaryPredictions
IUPL_PredictionScore_byProteins= IUPL_PredictionScore
IUPL_BinaryPrediction_byProteins= IUPL_BinaryPredictions


# # Featuers from SPOT

# In[29]:


NewScore10_byProtein= Spot_PredictionScore_byProteins
Spot_Prediction_ScoreWindow= [] 
newscore10window=0
b = 0
for b in range(0, len(NewScore10_byProtein),1):
                                                      if b == 0 : newscore10window= (NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/6
                                                      elif b == 1 : newscore10window= (NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/7
                                                      elif b == 2 : newscore10window= (NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/8
                                                      elif b == 3 : newscore10window= (NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/9
                                                      elif b == 4 : newscore10window= (NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/10
                                                      elif len(NewScore10_byProtein)-b == 5 : newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4])/10
                                                      elif len(NewScore10_byProtein)-b == 4 : newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3])/9
                                                      elif len(NewScore10_byProtein)-b == 3 : newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2])/8
                                                      elif len(NewScore10_byProtein)-b == 2 : newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1])/7
                                                      elif len(NewScore10_byProtein)-b == 1 : newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b])/6  
                                                      else: newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/11
                                                      Spot_Prediction_ScoreWindow.append(newscore10window)

Spot_Prediction_ScoreWindowWeighted= [] 
newscore10windowweighted=0
b = 0
for b in range(0, len(NewScore10_byProtein), 1):
                                                      if b == 0 : newscore10windowweighted= (NewScore10_byProtein[b]*0.65+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/6
                                                      elif b == 1 : newscore10windowweighted= (NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.56+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/7
                                                      elif b == 2 : newscore10windowweighted= (NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.48+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/8
                                                      elif b == 3 : newscore10windowweighted= (NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.41+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/9
                                                      elif b == 4 : newscore10windowweighted= (NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.35+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/10
                                                      elif len(NewScore10_byProtein)-b == 5 : newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.35+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06)/10
                                                      elif len(NewScore10_byProtein)-b == 4 : newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.41+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07)/9
                                                      elif len(NewScore10_byProtein)-b == 3 : newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.48+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08)/8
                                                      elif len(NewScore10_byProtein)-b == 2 : newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.56+NewScore10_byProtein[b+1]*0.09)/7
                                                      elif len(NewScore10_byProtein)-b == 1 : newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.65)/6  
                                                      else: newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.3+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/11
                                                      Spot_Prediction_ScoreWindowWeighted.append(newscore10windowweighted)
                                                    
Spot_Prediction_ScoreWindowDiff=[]
b=0
for b in range(0, len(NewScore10_byProtein), 1):
                                                      if b < 5 : extendedwindow= (NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9]+NewScore10_byProtein[b+10])/5
                                                      elif b == 5 : extendedwindow= (NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9]+NewScore10_byProtein[b+10])/6
                                                      elif b == 6 : extendedwindow= (NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9]+NewScore10_byProtein[b+10])/7
                                                      elif b == 7 : extendedwindow= (NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9]+NewScore10_byProtein[b+10])/8
                                                      elif b == 8 : extendedwindow= (NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9]+NewScore10_byProtein[b+10])/9
                                                      elif len(NewScore10_byProtein)-b < 7 : extendedwindow= (NewScore10_byProtein[b-10]+NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6])/5
                                                      elif len(NewScore10_byProtein)-b == 7 : extendedwindow= (NewScore10_byProtein[b-10]+NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6])/6
                                                      elif len(NewScore10_byProtein)-b == 8 : extendedwindow= (NewScore10_byProtein[b-10]+NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7])/7
                                                      elif len(NewScore10_byProtein)-b == 9 : extendedwindow= (NewScore10_byProtein[b-10]+NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8])/8
                                                      elif len(NewScore10_byProtein)-b == 10 : extendedwindow= (NewScore10_byProtein[b-10]+NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9])/9 
                                                      else: extendedwindow= (NewScore10_byProtein[b-10]+NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9]+NewScore10_byProtein[b+10])/10
                                                      Spot_Prediction_ScoreWindowDiff.append(extendedwindow)                                                    


# In[30]:


NewScore10_byProtein= Spot_BinaryPrediction_byProteins
Spot_Binary_PredictionWindow= [] 
newscore10window=0
b = 0
for b in range(0, len(NewScore10_byProtein), 1):
                                                      if b == 0 : newscore10window= (NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/6
                                                      elif b == 1 : newscore10window= (NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/7
                                                      elif b == 2 : newscore10window= (NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/8
                                                      elif b == 3 : newscore10window= (NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/9
                                                      elif b == 4 : newscore10window= (NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/10
                                                      elif len(NewScore10_byProtein)-b == 5 : newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4])/10
                                                      elif len(NewScore10_byProtein)-b == 4 : newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3])/9
                                                      elif len(NewScore10_byProtein)-b == 3 : newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2])/8
                                                      elif len(NewScore10_byProtein)-b == 2 : newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1])/7
                                                      elif len(NewScore10_byProtein)-b == 1 : newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b])/6  
                                                      else: newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/11
                                                      Spot_Binary_PredictionWindow.append(newscore10window)

Spot_Binary_PredictionWindowWeighted= [] 
newscore10windowweighted=0
b = 0
for b in range(0, len(NewScore10_byProtein), 1):
                                                      if b == 0 : newscore10windowweighted= (NewScore10_byProtein[b]*0.65+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/6
                                                      elif b == 1 : newscore10windowweighted= (NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.56+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/7
                                                      elif b == 2 : newscore10windowweighted= (NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.48+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/8
                                                      elif b == 3 : newscore10windowweighted= (NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.41+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/9
                                                      elif b == 4 : newscore10windowweighted= (NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.35+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/10
                                                      elif len(NewScore10_byProtein)-b == 5 : newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.35+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06)/10
                                                      elif len(NewScore10_byProtein)-b == 4 : newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.41+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07)/9
                                                      elif len(NewScore10_byProtein)-b == 3 : newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.48+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08)/8
                                                      elif len(NewScore10_byProtein)-b == 2 : newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.56+NewScore10_byProtein[b+1]*0.09)/7
                                                      elif len(NewScore10_byProtein)-b == 1 : newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.65)/6  
                                                      else: newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.3+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/11
                                                      Spot_Binary_PredictionWindowWeighted.append(newscore10windowweighted)
                                                    
Spot_Binary_PredictionWindowDiff=[]
b=0
for b in range(0, len(NewScore10_byProtein), 1):
                                                      if b < 5 : extendedwindow= (NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9]+NewScore10_byProtein[b+10])/5
                                                      elif b == 5 : extendedwindow= (NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9]+NewScore10_byProtein[b+10])/6
                                                      elif b == 6 : extendedwindow= (NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9]+NewScore10_byProtein[b+10])/7
                                                      elif b == 7 : extendedwindow= (NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9]+NewScore10_byProtein[b+10])/8
                                                      elif b == 8 : extendedwindow= (NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9]+NewScore10_byProtein[b+10])/9
                                                      elif len(NewScore10_byProtein)-b < 7 : extendedwindow= (NewScore10_byProtein[b-10]+NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6])/5
                                                      elif len(NewScore10_byProtein)-b == 7 : extendedwindow= (NewScore10_byProtein[b-10]+NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6])/6
                                                      elif len(NewScore10_byProtein)-b == 8 : extendedwindow= (NewScore10_byProtein[b-10]+NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7])/7
                                                      elif len(NewScore10_byProtein)-b == 9 : extendedwindow= (NewScore10_byProtein[b-10]+NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8])/8
                                                      elif len(NewScore10_byProtein)-b == 10 : extendedwindow= (NewScore10_byProtein[b-10]+NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9])/9 
                                                      else: extendedwindow= (NewScore10_byProtein[b-10]+NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9]+NewScore10_byProtein[b+10])/10
                                                      Spot_Binary_PredictionWindowDiff.append(extendedwindow)                                                    


# In[31]:


Score_byProtein = Spot_PredictionScore_byProteins
Spot_Prediction_Score_Up=[]
scoreup=0
b = 0
for b in range(0,len(Score_byProtein),1):
                                     if b == 0: scoreup =  Score_byProtein[0]
                                     else: scoreup =  Score_byProtein[b-1]
                                     Spot_Prediction_Score_Up.append(scoreup)
                                
Spot_Prediction_Score_Down=[]
scoredown=0
b = 0
for b in range(0,len(Score_byProtein),1):
                                     if len(Score_byProtein)-b == 1 : scoredown =  Score_byProtein[b]
                                     else: scoredown =  Score_byProtein[b+1]
                                     Spot_Prediction_Score_Down.append(scoredown)
                                
Spot_Prediction_Score_3AVG =[]
score3avg = 0
b = 0
for b in range(0,len(Score_byProtein),1):
                                     if b == 0: score3avg =  (Score_byProtein[b] + Score_byProtein[b+1] + Score_byProtein[b+1])/3
                                     elif len(Score_byProtein)-b == 1 : score3avg =  (Score_byProtein[b] + Score_byProtein[b-1] + Score_byProtein[b-1])/3
                                     else: score3avg =  (Score_byProtein[b-1] + Score_byProtein[b] + Score_byProtein[b+1])/3
                                     Spot_Prediction_Score_3AVG.append(score3avg)


# In[32]:


Score_byProtein = Spot_BinaryPrediction_byProteins
Spot_Binary_Prediction_Up=[]
scoreup=0
b = 0
for b in range(0,len(Score_byProtein),1):
                                     if b == 0: scoreup =  Score_byProtein[0]
                                     else: scoreup =  Score_byProtein[b-1]
                                     Spot_Binary_Prediction_Up.append(scoreup)
                                
Spot_Binary_Prediction_Down=[]
scoredown=0
b = 0
for b in range(0,len(Score_byProtein),1):
                                     if len(Score_byProtein)-b == 1 : scoredown =  Score_byProtein[b]
                                     else: scoredown =  Score_byProtein[b+1]
                                     Spot_Binary_Prediction_Down.append(scoredown)

Spot_Binary_Prediction_3AVG =[]
score3avg = 0
b = 0
for b in range(0,len(Score_byProtein),1):
                                     if b == 0: score3avg =  (Score_byProtein[b] + Score_byProtein[b+1] + Score_byProtein[b+1])/3
                                     elif len(Score_byProtein)-b == 1 : score3avg =  (Score_byProtein[b] + Score_byProtein[b-1] + Score_byProtein[b-1])/3
                                     else: score3avg =  (Score_byProtein[b-1] + Score_byProtein[b] + Score_byProtein[b+1])/3
                                     Spot_Binary_Prediction_3AVG.append(score3avg)


# In[33]:


#######################################################################################################
#Getting the positions of predicted Disorder residues
Disorder_Index= [] 
disorderindex=0
b = 0
for b in range(0, len(Spot_BinaryPredictions), 1):
                                         if Spot_BinaryPredictions[b] == 1 : disorderindex= b
                                         Disorder_Index.append(disorderindex)
Disorder_Index=np.unique(Disorder_Index)

######################################################################################################
#Getting the margins of disorder regions
Disorder_Margins= [] 
disordermargins=0
b = 0
h = len(Disorder_Index) -1
for b in range(0, h , 1):
                            if Disorder_Index[b+1]-Disorder_Index[b] != 1 : disordermargins= (b+1)
                            elif np.isin(Disorder_Index[b], d) == True : disordermargins= (b)
                            Disorder_Margins.append(disordermargins)
Disorder_Margins=np.unique(Disorder_Margins)

##############################################################################################################
#Splitting the disorder positions in to regions
Disorder_Regions= np.split(Disorder_Index,Disorder_Margins)

##############################################################################################################
#Calculating the depth of all disorder residues in their regons
Disorder_Depth= [] 
disorderdepth=0
b = 0
for Disorder_Region in Disorder_Regions:
                                for b in range(0, len(Disorder_Region), 1):
                                                      if b < (len(Disorder_Region))/2 : disorderdepth = b 
                                                      elif b > (len(Disorder_Region))/2 : disorderdepth = len(Disorder_Region) -(b+1) 
                                                      Disorder_Depth.append(disorderdepth)

################################################################################################################
#Assigning -1 to ordered residues
Disorder_DepthScore= [] 
disorderdepthscore=0
b = 0
for b in range(0,len(Spot_BinaryPredictions), 1):
                                         if Spot_BinaryPredictions[b] == 0 : disorderdepthscore= -1
                                         else: disorderdepthscore= Spot_BinaryPredictions[b]
                                         Disorder_DepthScore.append(disorderdepthscore)
                        
#############################################################################################################
#Casting lists to arrays 
Disorder_DepthScore = np.asarray(Disorder_DepthScore)
Disorder_Index = np.asarray(Disorder_Index)
Disorder_Depth = np.asarray(Disorder_Depth)

#Inserting depths to the position of disorder residues
np.put(Disorder_DepthScore, Disorder_Index, Disorder_Depth)
Spot_Disorder_DepthScore=list(Disorder_DepthScore)


# In[34]:


#######################################################################################################
#Getting the positions of predicted Order regions
Order_Index= [] 
orderindex=0
b = 0
for b in range(0, len(Spot_BinaryPredictions), 1):
                                         if Spot_BinaryPredictions[b] == 0 : orderindex= b
                                         Order_Index.append(orderindex)
Order_Index=np.unique(Order_Index)

######################################################################################################
#Getting the margins of disorder regions
Order_Margins= [] 
ordermargins=0
b = 0
h = len(Order_Index) -1
for b in range(0,h , 1):
                            if Order_Index[b+1]-Order_Index[b] != 1 : ordermargins= (b+1)
                            elif np.isin(Order_Index[b], d) == True : ordermargins= (b)
                            Order_Margins.append(ordermargins)
Order_Margins=np.unique(Order_Margins)

#############################################################################################################
#Splitting the disorder positions in to regions
Order_Regions= np.split(Order_Index,Order_Margins)

#############################################################################################################
#Calculating the depth of all disorder residues in their regons
Order_Depth= [] 
orderdepth=0
b = 0
for Order_Region in Order_Regions:
                                for b in range(0, len(Order_Region), 1):
                                                      if b < (len(Order_Region))/2 : orderdepth = b 
                                                      elif b > (len(Order_Region))/2 : orderdepth = len(Order_Region) -(b+1)
                                                      Order_Depth.append(orderdepth)

################################################################################################################
#Assigning -1 to disordered residues
Order_DepthScore= [] 
orderdepthscore=0
b = 0
for b in range(0, len(Spot_BinaryPredictions), 1):
                                         if Spot_BinaryPredictions[b] == 1 : orderdepthscore= -1
                                         else: orderdepthscore= Spot_BinaryPredictions[b]
                                         Order_DepthScore.append(orderdepthscore)
                        
#############################################################################################################
#Casting lists to arrays 
Order_DepthScore = np.asarray(Order_DepthScore)
Order_Index = np.asarray(Order_Index)
Order_Depth = np.asarray(Order_Depth)

#Inserting depths to the position of disorder residues
np.put(Order_DepthScore, Order_Index, Order_Depth)
Spot_Order_DepthScore=list(Order_DepthScore)


# # Featuers from IUPL

# In[35]:


NewScore10_byProtein= IUPL_PredictionScore_byProteins
IUPL_Prediction_ScoreWindow= [] 
newscore10window=0
b = 0
for b in range(0, len(NewScore10_byProtein), 1):
                                                      if b == 0 : newscore10window= (NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/6
                                                      elif b == 1 : newscore10window= (NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/7
                                                      elif b == 2 : newscore10window= (NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/8
                                                      elif b == 3 : newscore10window= (NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/9
                                                      elif b == 4 : newscore10window= (NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/10
                                                      elif len(NewScore10_byProtein)-b == 5 : newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4])/10
                                                      elif len(NewScore10_byProtein)-b == 4 : newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3])/9
                                                      elif len(NewScore10_byProtein)-b == 3 : newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2])/8
                                                      elif len(NewScore10_byProtein)-b == 2 : newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1])/7
                                                      elif len(NewScore10_byProtein)-b == 1 : newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b])/6  
                                                      else: newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/11
                                                      IUPL_Prediction_ScoreWindow.append(newscore10window)

IUPL_Prediction_ScoreWindowWeighted= [] 
newscore10windowweighted=0
b = 0
for b in range(0, len(NewScore10_byProtein), 1):
                                                      if b == 0 : newscore10windowweighted= (NewScore10_byProtein[b]*0.65+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/6
                                                      elif b == 1 : newscore10windowweighted= (NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.56+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/7
                                                      elif b == 2 : newscore10windowweighted= (NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.48+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/8
                                                      elif b == 3 : newscore10windowweighted= (NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.41+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/9
                                                      elif b == 4 : newscore10windowweighted= (NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.35+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/10
                                                      elif len(NewScore10_byProtein)-b == 5 : newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.35+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06)/10
                                                      elif len(NewScore10_byProtein)-b == 4 : newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.41+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07)/9
                                                      elif len(NewScore10_byProtein)-b == 3 : newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.48+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08)/8
                                                      elif len(NewScore10_byProtein)-b == 2 : newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.56+NewScore10_byProtein[b+1]*0.09)/7
                                                      elif len(NewScore10_byProtein)-b == 1 : newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.65)/6  
                                                      else: newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.3+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/11
                                                      IUPL_Prediction_ScoreWindowWeighted.append(newscore10windowweighted)
                                                    
IUPL_Prediction_ScoreWindowDiff=[]
b=0
for b in range(0, len(NewScore10_byProtein), 1):
                                                      if b < 5 : extendedwindow= (NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9]+NewScore10_byProtein[b+10])/5
                                                      elif b == 5 : extendedwindow= (NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9]+NewScore10_byProtein[b+10])/6
                                                      elif b == 6 : extendedwindow= (NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9]+NewScore10_byProtein[b+10])/7
                                                      elif b == 7 : extendedwindow= (NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9]+NewScore10_byProtein[b+10])/8
                                                      elif b == 8 : extendedwindow= (NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9]+NewScore10_byProtein[b+10])/9
                                                      elif len(NewScore10_byProtein)-b < 7 : extendedwindow= (NewScore10_byProtein[b-10]+NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6])/5
                                                      elif len(NewScore10_byProtein)-b == 7 : extendedwindow= (NewScore10_byProtein[b-10]+NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6])/6
                                                      elif len(NewScore10_byProtein)-b == 8 : extendedwindow= (NewScore10_byProtein[b-10]+NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7])/7
                                                      elif len(NewScore10_byProtein)-b == 9 : extendedwindow= (NewScore10_byProtein[b-10]+NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8])/8
                                                      elif len(NewScore10_byProtein)-b == 10 : extendedwindow= (NewScore10_byProtein[b-10]+NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9])/9 
                                                      else: extendedwindow= (NewScore10_byProtein[b-10]+NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9]+NewScore10_byProtein[b+10])/10
                                                      IUPL_Prediction_ScoreWindowDiff.append(extendedwindow)                                                    


# In[36]:


NewScore10_byProtein= IUPL_BinaryPrediction_byProteins
IUPL_Binary_PredictionWindow= [] 
newscore10window=0
b = 0
for b in range(0, len(NewScore10_byProtein), 1):
                                                      if b == 0 : newscore10window= (NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/6
                                                      elif b == 1 : newscore10window= (NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/7
                                                      elif b == 2 : newscore10window= (NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/8
                                                      elif b == 3 : newscore10window= (NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/9
                                                      elif b == 4 : newscore10window= (NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/10
                                                      elif len(NewScore10_byProtein)-b == 5 : newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4])/10
                                                      elif len(NewScore10_byProtein)-b == 4 : newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3])/9
                                                      elif len(NewScore10_byProtein)-b == 3 : newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2])/8
                                                      elif len(NewScore10_byProtein)-b == 2 : newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1])/7
                                                      elif len(NewScore10_byProtein)-b == 1 : newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b])/6  
                                                      else: newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/11
                                                      IUPL_Binary_PredictionWindow.append(newscore10window)

IUPL_Binary_PredictionWindowWeighted= [] 
newscore10windowweighted=0
b = 0
for b in range(0, len(NewScore10_byProtein), 1):
                                                      if b == 0 : newscore10windowweighted= (NewScore10_byProtein[b]*0.65+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/6
                                                      elif b == 1 : newscore10windowweighted= (NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.56+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/7
                                                      elif b == 2 : newscore10windowweighted= (NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.48+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/8
                                                      elif b == 3 : newscore10windowweighted= (NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.41+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/9
                                                      elif b == 4 : newscore10windowweighted= (NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.35+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/10
                                                      elif len(NewScore10_byProtein)-b == 5 : newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.35+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06)/10
                                                      elif len(NewScore10_byProtein)-b == 4 : newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.41+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07)/9
                                                      elif len(NewScore10_byProtein)-b == 3 : newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.48+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08)/8
                                                      elif len(NewScore10_byProtein)-b == 2 : newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.56+NewScore10_byProtein[b+1]*0.09)/7
                                                      elif len(NewScore10_byProtein)-b == 1 : newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.65)/6  
                                                      else: newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.3+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/11
                                                      IUPL_Binary_PredictionWindowWeighted.append(newscore10windowweighted)
                                                    
IUPL_Binary_PredictionWindowDiff=[]
b=0
for b in range(0, len(NewScore10_byProtein), 1):
                                                      if b < 5 : extendedwindow= (NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9]+NewScore10_byProtein[b+10])/5
                                                      elif b == 5 : extendedwindow= (NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9]+NewScore10_byProtein[b+10])/6
                                                      elif b == 6 : extendedwindow= (NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9]+NewScore10_byProtein[b+10])/7
                                                      elif b == 7 : extendedwindow= (NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9]+NewScore10_byProtein[b+10])/8
                                                      elif b == 8 : extendedwindow= (NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9]+NewScore10_byProtein[b+10])/9
                                                      elif len(NewScore10_byProtein)-b < 7 : extendedwindow= (NewScore10_byProtein[b-10]+NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6])/5
                                                      elif len(NewScore10_byProtein)-b == 7 : extendedwindow= (NewScore10_byProtein[b-10]+NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6])/6
                                                      elif len(NewScore10_byProtein)-b == 8 : extendedwindow= (NewScore10_byProtein[b-10]+NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7])/7
                                                      elif len(NewScore10_byProtein)-b == 9 : extendedwindow= (NewScore10_byProtein[b-10]+NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8])/8
                                                      elif len(NewScore10_byProtein)-b == 10 : extendedwindow= (NewScore10_byProtein[b-10]+NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9])/9 
                                                      else: extendedwindow= (NewScore10_byProtein[b-10]+NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9]+NewScore10_byProtein[b+10])/10
                                                      IUPL_Binary_PredictionWindowDiff.append(extendedwindow)


# In[37]:


Score_byProtein = IUPL_PredictionScore_byProteins
IUPL_Prediction_Score_Up=[]
scoreup=0
b = 0
for b in range(0,len(Score_byProtein),1):
                                     if b == 0: scoreup =  Score_byProtein[0]
                                     else: scoreup =  Score_byProtein[b-1]
                                     IUPL_Prediction_Score_Up.append(scoreup)
                                
IUPL_Prediction_Score_Down=[]
scoredown=0
b = 0
for b in range(0,len(Score_byProtein),1):
                                     if len(Score_byProtein)-b == 1 : scoredown =  Score_byProtein[b]
                                     else: scoredown =  Score_byProtein[b+1]
                                     IUPL_Prediction_Score_Down.append(scoredown)
                                
IUPL_Prediction_Score_3AVG =[]
score3avg = 0
b = 0
for b in range(0,len(Score_byProtein),1):
                                     if b == 0: score3avg =  (Score_byProtein[b] + Score_byProtein[b+1] + Score_byProtein[b+1])/3
                                     elif len(Score_byProtein)-b == 1 : score3avg =  (Score_byProtein[b] + Score_byProtein[b-1] + Score_byProtein[b-1])/3
                                     else: score3avg =  (Score_byProtein[b-1] + Score_byProtein[b] + Score_byProtein[b+1])/3
                                     IUPL_Prediction_Score_3AVG.append(score3avg)


# In[38]:


Score_byProtein = IUPL_BinaryPrediction_byProteins
IUPL_Binary_Prediction_Up=[]
scoreup=0
b = 0
for b in range(0,len(Score_byProtein),1):
                                     if b == 0: scoreup =  Score_byProtein[0]
                                     else: scoreup =  Score_byProtein[b-1]
                                     IUPL_Binary_Prediction_Up.append(scoreup)
                                
IUPL_Binary_Prediction_Down=[]
scoredown=0
b = 0
for b in range(0,len(Score_byProtein),1):
                                     if len(Score_byProtein)-b == 1 : scoredown =  Score_byProtein[b]
                                     else: scoredown =  Score_byProtein[b+1]
                                     IUPL_Binary_Prediction_Down.append(scoredown)

IUPL_Binary_Prediction_3AVG =[]
score3avg = 0
b = 0
for b in range(0,len(Score_byProtein),1):
                                     if b == 0: score3avg =  (Score_byProtein[b] + Score_byProtein[b+1] + Score_byProtein[b+1])/3
                                     elif len(Score_byProtein)-b == 1 : score3avg =  (Score_byProtein[b] + Score_byProtein[b-1] + Score_byProtein[b-1])/3
                                     else: score3avg =  (Score_byProtein[b-1] + Score_byProtein[b] + Score_byProtein[b+1])/3
                                     IUPL_Binary_Prediction_3AVG.append(score3avg)


# In[39]:


#######################################################################################################
#Getting the positions of predicted Disorder residues
Disorder_Index= [] 
disorderindex=0
b = 0
for b in range(0, len(IUPL_BinaryPredictions), 1):
                                         if IUPL_BinaryPredictions[b] == 1 : disorderindex= b
                                         Disorder_Index.append(disorderindex)
Disorder_Index=np.unique(Disorder_Index)

######################################################################################################
#Getting the margins of disorder regions
Disorder_Margins= [] 
disordermargins=0
b = 0
h = len(Disorder_Index) -1
for b in range(0, h , 1):
                            if Disorder_Index[b+1]-Disorder_Index[b] != 1 : disordermargins= (b+1)
                            elif np.isin(Disorder_Index[b], d) == True : disordermargins= (b)
                            Disorder_Margins.append(disordermargins)
Disorder_Margins=np.unique(Disorder_Margins)

##############################################################################################################
#Splitting the disorder positions in to regions
Disorder_Regions= np.split(Disorder_Index,Disorder_Margins)

##############################################################################################################
#Calculating the depth of all disorder residues in their regons
Disorder_Depth= [] 
disorderdepth=0
b = 0
for Disorder_Region in Disorder_Regions:
                                for b in range(0, len(Disorder_Region), 1):
                                                      if b < (len(Disorder_Region))/2 : disorderdepth = b 
                                                      elif b > (len(Disorder_Region))/2 : disorderdepth = len(Disorder_Region) -(b+1) 
                                                      Disorder_Depth.append(disorderdepth)

################################################################################################################
#Assigning -1 to ordered residues
Disorder_DepthScore= [] 
disorderdepthscore=0
b = 0
for b in range(0, len(IUPL_BinaryPredictions), 1):
                                         if IUPL_BinaryPredictions[b] == 0 : disorderdepthscore= -1
                                         else: disorderdepthscore= IUPL_BinaryPredictions[b]
                                         Disorder_DepthScore.append(disorderdepthscore)
                        
#############################################################################################################
#Casting lists to arrays 
Disorder_DepthScore = np.asarray(Disorder_DepthScore)
Disorder_Index = np.asarray(Disorder_Index)
Disorder_Depth = np.asarray(Disorder_Depth)

#Inserting depths to the position of disorder residues
np.put(Disorder_DepthScore, Disorder_Index, Disorder_Depth)
IUPL_Disorder_DepthScore=list(Disorder_DepthScore)


# In[40]:


#######################################################################################################
#Getting the positions of predicted Order regions
Order_Index= [] 
orderindex=0
b = 0
for b in range(0,len(IUPL_BinaryPredictions),1):
                                         if IUPL_BinaryPredictions[b] == 0 : orderindex= b
                                         Order_Index.append(orderindex)
Order_Index=np.unique(Order_Index)

######################################################################################################
#Getting the margins of disorder regions
Order_Margins= [] 
ordermargins=0
b = 0
h = len(Order_Index) -1
for b in range(0,h , 1):
                            if Order_Index[b+1]-Order_Index[b] != 1 : ordermargins= (b+1)
                            elif np.isin(Order_Index[b], d) == True : ordermargins= (b)
                            Order_Margins.append(ordermargins)
Order_Margins=np.unique(Order_Margins)

#############################################################################################################
#Splitting the disorder positions in to regions
Order_Regions= np.split(Order_Index,Order_Margins)

#############################################################################################################
#Calculating the depth of all disorder residues in their regons
Order_Depth= [] 
orderdepth=0
b = 0
for Order_Region in Order_Regions:
                                for b in range(0, len(Order_Region), 1):
                                                      if b < (len(Order_Region))/2 : orderdepth = b 
                                                      elif b > (len(Order_Region))/2 : orderdepth = len(Order_Region) -(b+1)
                                                      Order_Depth.append(orderdepth)

################################################################################################################
#Assigning -1 to disordered residues
Order_DepthScore= [] 
orderdepthscore=0
b = 0
for b in range(0, len(IUPL_BinaryPredictions),1):
                                         if IUPL_BinaryPredictions[b] == 1 : orderdepthscore= -1
                                         else: orderdepthscore= IUPL_BinaryPredictions[b]
                                         Order_DepthScore.append(orderdepthscore)
                        
#############################################################################################################
#Casting lists to arrays 
Order_DepthScore = np.asarray(Order_DepthScore)
Order_Index = np.asarray(Order_Index)
Order_Depth = np.asarray(Order_Depth)

#Inserting depths to the position of disorder residues
np.put(Order_DepthScore, Order_Index, Order_Depth)
IUPL_Order_DepthScore=list(Order_DepthScore)


# # Combined featuers of IUPL and SPOT

# In[41]:


Consensus_Score=[]
b = 0
for b in range(0,len(Spot_BinaryPredictions),1):
    if IUPL_BinaryPredictions[b]== Spot_BinaryPredictions[b]and Spot_BinaryPredictions[b]==1:conscore=np.fmax(IUPL_PredictionScore[b],Spot_PredictionScore[b])
    elif IUPL_BinaryPredictions[b]!= Spot_BinaryPredictions[b]and Spot_BinaryPredictions[b]==0:conscore=np.fmin(IUPL_PredictionScore[b],Spot_PredictionScore[b])
    else:conscore =Spot_PredictionScore[b]
    Consensus_Score.append(conscore) 
Consensus_Score_byProteins= Consensus_Score


# In[42]:


NewScore10_byProtein= Consensus_Score_byProteins
Consensus_ScoreWindow= [] 
newscore10window=0
b = 0
for b in range(0, len(NewScore10_byProtein), 1):
                                                      if b == 0 : newscore10window= (NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/6
                                                      elif b == 1 : newscore10window= (NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/7
                                                      elif b == 2 : newscore10window= (NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/8
                                                      elif b == 3 : newscore10window= (NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/9
                                                      elif b == 4 : newscore10window= (NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/10
                                                      elif len(NewScore10_byProtein)-b == 5 : newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4])/10
                                                      elif len(NewScore10_byProtein)-b == 4 : newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3])/9
                                                      elif len(NewScore10_byProtein)-b == 3 : newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2])/8
                                                      elif len(NewScore10_byProtein)-b == 2 : newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1])/7
                                                      elif len(NewScore10_byProtein)-b == 1 : newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b])/6  
                                                      else: newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/11
                                                      Consensus_ScoreWindow.append(newscore10window)

Consensus_ScoreWindowWeighted= [] 
newscore10windowweighted=0
b = 0
for b in range(0, len(NewScore10_byProtein), 1):
                                                      if b == 0 : newscore10windowweighted= (NewScore10_byProtein[b]*0.65+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/6
                                                      elif b == 1 : newscore10windowweighted= (NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.56+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/7
                                                      elif b == 2 : newscore10windowweighted= (NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.48+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/8
                                                      elif b == 3 : newscore10windowweighted= (NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.41+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/9
                                                      elif b == 4 : newscore10windowweighted= (NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.35+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/10
                                                      elif len(NewScore10_byProtein)-b == 5 : newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.35+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06)/10
                                                      elif len(NewScore10_byProtein)-b == 4 : newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.41+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07)/9
                                                      elif len(NewScore10_byProtein)-b == 3 : newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.48+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08)/8
                                                      elif len(NewScore10_byProtein)-b == 2 : newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.56+NewScore10_byProtein[b+1]*0.09)/7
                                                      elif len(NewScore10_byProtein)-b == 1 : newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.65)/6  
                                                      else: newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.3+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/11
                                                      Consensus_ScoreWindowWeighted.append(newscore10windowweighted)
                                                    
Consensus_ScoreWindowDiff=[]
b=0
for b in range(0, len(NewScore10_byProtein), 1):
                                                      if b < 5 : extendedwindow= (NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9]+NewScore10_byProtein[b+10])/5
                                                      elif b == 5 : extendedwindow= (NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9]+NewScore10_byProtein[b+10])/6
                                                      elif b == 6 : extendedwindow= (NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9]+NewScore10_byProtein[b+10])/7
                                                      elif b == 7 : extendedwindow= (NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9]+NewScore10_byProtein[b+10])/8
                                                      elif b == 8 : extendedwindow= (NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9]+NewScore10_byProtein[b+10])/9
                                                      elif len(NewScore10_byProtein)-b < 7 : extendedwindow= (NewScore10_byProtein[b-10]+NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6])/5
                                                      elif len(NewScore10_byProtein)-b == 7 : extendedwindow= (NewScore10_byProtein[b-10]+NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6])/6
                                                      elif len(NewScore10_byProtein)-b == 8 : extendedwindow= (NewScore10_byProtein[b-10]+NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7])/7
                                                      elif len(NewScore10_byProtein)-b == 9 : extendedwindow= (NewScore10_byProtein[b-10]+NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8])/8
                                                      elif len(NewScore10_byProtein)-b == 10 : extendedwindow= (NewScore10_byProtein[b-10]+NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9])/9 
                                                      else: extendedwindow= (NewScore10_byProtein[b-10]+NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9]+NewScore10_byProtein[b+10])/10
                                                      Consensus_ScoreWindowDiff.append(extendedwindow)


# In[43]:


#Product of Prediction Scores
New_Score1=[]
b=0
for b in range(0,len(Spot_PredictionScore),1):
                    new_score= Spot_PredictionScore[b]*IUPL_PredictionScore[b]
                    New_Score1.append(new_score)  
New_Score1_byProteins= New_Score1


# In[44]:


NewScore10_byProtein= New_Score1_byProteins
New_Score1Window= [] 
newscore10window=0
b = 0
for b in range(0, len(NewScore10_byProtein), 1):
                                                      if b == 0 : newscore10window= (NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/6
                                                      elif b == 1 : newscore10window= (NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/7
                                                      elif b == 2 : newscore10window= (NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/8
                                                      elif b == 3 : newscore10window= (NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/9
                                                      elif b == 4 : newscore10window= (NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/10
                                                      elif len(NewScore10_byProtein)-b == 5 : newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4])/10
                                                      elif len(NewScore10_byProtein)-b == 4 : newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3])/9
                                                      elif len(NewScore10_byProtein)-b == 3 : newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2])/8
                                                      elif len(NewScore10_byProtein)-b == 2 : newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1])/7
                                                      elif len(NewScore10_byProtein)-b == 1 : newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b])/6  
                                                      else: newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/11
                                                      New_Score1Window.append(newscore10window)

New_Score1WindowWeighted= [] 
newscore10windowweighted=0
b = 0
for b in range(0, len(NewScore10_byProtein), 1):
                                                      if b == 0 : newscore10windowweighted= (NewScore10_byProtein[b]*0.65+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/6
                                                      elif b == 1 : newscore10windowweighted= (NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.56+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/7
                                                      elif b == 2 : newscore10windowweighted= (NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.48+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/8
                                                      elif b == 3 : newscore10windowweighted= (NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.41+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/9
                                                      elif b == 4 : newscore10windowweighted= (NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.35+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/10
                                                      elif len(NewScore10_byProtein)-b == 5 : newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.35+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06)/10
                                                      elif len(NewScore10_byProtein)-b == 4 : newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.41+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07)/9
                                                      elif len(NewScore10_byProtein)-b == 3 : newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.48+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08)/8
                                                      elif len(NewScore10_byProtein)-b == 2 : newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.56+NewScore10_byProtein[b+1]*0.09)/7
                                                      elif len(NewScore10_byProtein)-b == 1 : newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.65)/6  
                                                      else: newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.3+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/11
                                                      New_Score1WindowWeighted.append(newscore10windowweighted)
                                                    
New_Score1WindowDiff=[]
b=0
for b in range(0, len(NewScore10_byProtein), 1):
                                                      if b < 5 : extendedwindow= (NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9]+NewScore10_byProtein[b+10])/5
                                                      elif b == 5 : extendedwindow= (NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9]+NewScore10_byProtein[b+10])/6
                                                      elif b == 6 : extendedwindow= (NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9]+NewScore10_byProtein[b+10])/7
                                                      elif b == 7 : extendedwindow= (NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9]+NewScore10_byProtein[b+10])/8
                                                      elif b == 8 : extendedwindow= (NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9]+NewScore10_byProtein[b+10])/9
                                                      elif len(NewScore10_byProtein)-b < 7 : extendedwindow= (NewScore10_byProtein[b-10]+NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6])/5
                                                      elif len(NewScore10_byProtein)-b == 7 : extendedwindow= (NewScore10_byProtein[b-10]+NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6])/6
                                                      elif len(NewScore10_byProtein)-b == 8 : extendedwindow= (NewScore10_byProtein[b-10]+NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7])/7
                                                      elif len(NewScore10_byProtein)-b == 9 : extendedwindow= (NewScore10_byProtein[b-10]+NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8])/8
                                                      elif len(NewScore10_byProtein)-b == 10 : extendedwindow= (NewScore10_byProtein[b-10]+NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9])/9 
                                                      else: extendedwindow= (NewScore10_byProtein[b-10]+NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9]+NewScore10_byProtein[b+10])/10
                                                      New_Score1WindowDiff.append(extendedwindow)


# In[45]:


#Consensus division and product
New_Score2=[]
b=0
for b in range(0,len(Spot_PredictionScore),1):
    if IUPL_BinaryPredictions[b]== Spot_BinaryPredictions[b]and Spot_BinaryPredictions[b]==1:new_score=(Spot_PredictionScore[b]/IUPL_PredictionScore[b])
    elif IUPL_BinaryPredictions[b]!= Spot_BinaryPredictions[b]and Spot_BinaryPredictions[b]==0:new_score=(IUPL_PredictionScore[b]*Spot_PredictionScore[b])
    else: new_score=Spot_PredictionScore[b]
    New_Score2.append(new_score)
New_Score2_byProteins= New_Score2


# In[46]:


NewScore10_byProtein= New_Score2_byProteins
New_Score2Window= [] 
newscore10window=0
b = 0
for b in range(0, len(NewScore10_byProtein), 1):
                                                      if b == 0 : newscore10window= (NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/6
                                                      elif b == 1 : newscore10window= (NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/7
                                                      elif b == 2 : newscore10window= (NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/8
                                                      elif b == 3 : newscore10window= (NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/9
                                                      elif b == 4 : newscore10window= (NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/10
                                                      elif len(NewScore10_byProtein)-b == 5 : newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4])/10
                                                      elif len(NewScore10_byProtein)-b == 4 : newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3])/9
                                                      elif len(NewScore10_byProtein)-b == 3 : newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2])/8
                                                      elif len(NewScore10_byProtein)-b == 2 : newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1])/7
                                                      elif len(NewScore10_byProtein)-b == 1 : newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b])/6  
                                                      else: newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/11
                                                      New_Score2Window.append(newscore10window)

New_Score2WindowWeighted= [] 
newscore10windowweighted=0
b = 0
for b in range(0, len(NewScore10_byProtein), 1):
                                                      if b == 0 : newscore10windowweighted= (NewScore10_byProtein[b]*0.65+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/6
                                                      elif b == 1 : newscore10windowweighted= (NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.56+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/7
                                                      elif b == 2 : newscore10windowweighted= (NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.48+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/8
                                                      elif b == 3 : newscore10windowweighted= (NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.41+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/9
                                                      elif b == 4 : newscore10windowweighted= (NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.35+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/10
                                                      elif len(NewScore10_byProtein)-b == 5 : newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.35+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06)/10
                                                      elif len(NewScore10_byProtein)-b == 4 : newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.41+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07)/9
                                                      elif len(NewScore10_byProtein)-b == 3 : newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.48+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08)/8
                                                      elif len(NewScore10_byProtein)-b == 2 : newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.56+NewScore10_byProtein[b+1]*0.09)/7
                                                      elif len(NewScore10_byProtein)-b == 1 : newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.65)/6  
                                                      else: newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.3+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/11
                                                      New_Score2WindowWeighted.append(newscore10windowweighted)
                                                    
New_Score2WindowDiff=[]
b=0
for b in range(0, len(NewScore10_byProtein), 1):
                                                      if b < 5 : extendedwindow= (NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9]+NewScore10_byProtein[b+10])/5
                                                      elif b == 5 : extendedwindow= (NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9]+NewScore10_byProtein[b+10])/6
                                                      elif b == 6 : extendedwindow= (NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9]+NewScore10_byProtein[b+10])/7
                                                      elif b == 7 : extendedwindow= (NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9]+NewScore10_byProtein[b+10])/8
                                                      elif b == 8 : extendedwindow= (NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9]+NewScore10_byProtein[b+10])/9
                                                      elif len(NewScore10_byProtein)-b < 7 : extendedwindow= (NewScore10_byProtein[b-10]+NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6])/5
                                                      elif len(NewScore10_byProtein)-b == 7 : extendedwindow= (NewScore10_byProtein[b-10]+NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6])/6
                                                      elif len(NewScore10_byProtein)-b == 8 : extendedwindow= (NewScore10_byProtein[b-10]+NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7])/7
                                                      elif len(NewScore10_byProtein)-b == 9 : extendedwindow= (NewScore10_byProtein[b-10]+NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8])/8
                                                      elif len(NewScore10_byProtein)-b == 10 : extendedwindow= (NewScore10_byProtein[b-10]+NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9])/9 
                                                      else: extendedwindow= (NewScore10_byProtein[b-10]+NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9]+NewScore10_byProtein[b+10])/10
                                                      New_Score2WindowDiff.append(extendedwindow)


# In[47]:


# Spot score optimization
New_Score3=[]
b=0
for b in range(0,len(Spot_PredictionScore),1):
    if Spot_BinaryPredictions[b]==1:new_score=(IUPL_PredictionScore[b]/Spot_PredictionScore[b])
    if Spot_BinaryPredictions[b]==0:new_score=(IUPL_PredictionScore[b]*Spot_PredictionScore[b])
    New_Score3.append(new_score)
New_Score3_byProteins= New_Score3


# In[48]:


NewScore10_byProtein= New_Score3_byProteins
New_Score3Window= [] 
newscore10window=0
b = 0
for b in range(0, len(NewScore10_byProtein), 1):
                                                      if b == 0 : newscore10window= (NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/6
                                                      elif b == 1 : newscore10window= (NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/7
                                                      elif b == 2 : newscore10window= (NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/8
                                                      elif b == 3 : newscore10window= (NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/9
                                                      elif b == 4 : newscore10window= (NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/10
                                                      elif len(NewScore10_byProtein)-b == 5 : newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4])/10
                                                      elif len(NewScore10_byProtein)-b == 4 : newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3])/9
                                                      elif len(NewScore10_byProtein)-b == 3 : newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2])/8
                                                      elif len(NewScore10_byProtein)-b == 2 : newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1])/7
                                                      elif len(NewScore10_byProtein)-b == 1 : newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b])/6  
                                                      else: newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/11
                                                      New_Score3Window.append(newscore10window)

New_Score3WindowWeighted= [] 
newscore10windowweighted=0
b = 0
for b in range(0, len(NewScore10_byProtein), 1):
                                                      if b == 0 : newscore10windowweighted= (NewScore10_byProtein[b]*0.65+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/6
                                                      elif b == 1 : newscore10windowweighted= (NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.56+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/7
                                                      elif b == 2 : newscore10windowweighted= (NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.48+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/8
                                                      elif b == 3 : newscore10windowweighted= (NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.41+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/9
                                                      elif b == 4 : newscore10windowweighted= (NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.35+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/10
                                                      elif len(NewScore10_byProtein)-b == 5 : newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.35+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06)/10
                                                      elif len(NewScore10_byProtein)-b == 4 : newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.41+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07)/9
                                                      elif len(NewScore10_byProtein)-b == 3 : newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.48+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08)/8
                                                      elif len(NewScore10_byProtein)-b == 2 : newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.56+NewScore10_byProtein[b+1]*0.09)/7
                                                      elif len(NewScore10_byProtein)-b == 1 : newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.65)/6  
                                                      else: newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.3+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/11
                                                      New_Score3WindowWeighted.append(newscore10windowweighted)
                                                    
New_Score3WindowDiff=[]
b=0
for b in range(0, len(NewScore10_byProtein), 1):
                                                      if b < 5 : extendedwindow= (NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9]+NewScore10_byProtein[b+10])/5
                                                      elif b == 5 : extendedwindow= (NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9]+NewScore10_byProtein[b+10])/6
                                                      elif b == 6 : extendedwindow= (NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9]+NewScore10_byProtein[b+10])/7
                                                      elif b == 7 : extendedwindow= (NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9]+NewScore10_byProtein[b+10])/8
                                                      elif b == 8 : extendedwindow= (NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9]+NewScore10_byProtein[b+10])/9
                                                      elif len(NewScore10_byProtein)-b < 7 : extendedwindow= (NewScore10_byProtein[b-10]+NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6])/5
                                                      elif len(NewScore10_byProtein)-b == 7 : extendedwindow= (NewScore10_byProtein[b-10]+NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6])/6
                                                      elif len(NewScore10_byProtein)-b == 8 : extendedwindow= (NewScore10_byProtein[b-10]+NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7])/7
                                                      elif len(NewScore10_byProtein)-b == 9 : extendedwindow= (NewScore10_byProtein[b-10]+NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8])/8
                                                      elif len(NewScore10_byProtein)-b == 10 : extendedwindow= (NewScore10_byProtein[b-10]+NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9])/9 
                                                      else: extendedwindow= (NewScore10_byProtein[b-10]+NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9]+NewScore10_byProtein[b+10])/10
                                                      New_Score3WindowDiff.append(extendedwindow)


# In[49]:


#Product of Binary
New_Binary1=[]
b=0
for b in range(0,len(Spot_BinaryPredictions),1):
               new_binary=Spot_BinaryPredictions[b]*IUPL_BinaryPredictions[b]
               New_Binary1.append(new_binary)
New_Binary1_byProteins= New_Binary1


# In[50]:


NewScore10_byProtein= New_Binary1_byProteins
New_Binary1Window= [] 
newscore10window=0
b = 0
for b in range(0, len(NewScore10_byProtein), 1):
                                                      if b == 0 : newscore10window= (NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/6
                                                      elif b == 1 : newscore10window= (NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/7
                                                      elif b == 2 : newscore10window= (NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/8
                                                      elif b == 3 : newscore10window= (NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/9
                                                      elif b == 4 : newscore10window= (NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/10
                                                      elif len(NewScore10_byProtein)-b == 5 : newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4])/10
                                                      elif len(NewScore10_byProtein)-b == 4 : newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3])/9
                                                      elif len(NewScore10_byProtein)-b == 3 : newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2])/8
                                                      elif len(NewScore10_byProtein)-b == 2 : newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1])/7
                                                      elif len(NewScore10_byProtein)-b == 1 : newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b])/6  
                                                      else: newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/11
                                                      New_Binary1Window.append(newscore10window)

New_Binary1WindowWeighted= [] 
newscore10windowweighted=0
b = 0
for b in range(0, len(NewScore10_byProtein), 1):
                                                      if b == 0 : newscore10windowweighted= (NewScore10_byProtein[b]*0.65+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/6
                                                      elif b == 1 : newscore10windowweighted= (NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.56+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/7
                                                      elif b == 2 : newscore10windowweighted= (NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.48+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/8
                                                      elif b == 3 : newscore10windowweighted= (NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.41+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/9
                                                      elif b == 4 : newscore10windowweighted= (NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.35+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/10
                                                      elif len(NewScore10_byProtein)-b == 5 : newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.35+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06)/10
                                                      elif len(NewScore10_byProtein)-b == 4 : newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.41+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07)/9
                                                      elif len(NewScore10_byProtein)-b == 3 : newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.48+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08)/8
                                                      elif len(NewScore10_byProtein)-b == 2 : newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.56+NewScore10_byProtein[b+1]*0.09)/7
                                                      elif len(NewScore10_byProtein)-b == 1 : newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.65)/6  
                                                      else: newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.3+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/11
                                                      New_Binary1WindowWeighted.append(newscore10windowweighted)
                                                    
New_Binary1WindowDiff=[]
b=0
for b in range(0, len(NewScore10_byProtein), 1):
                                                      if b < 5 : extendedwindow= (NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9]+NewScore10_byProtein[b+10])/5
                                                      elif b == 5 : extendedwindow= (NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9]+NewScore10_byProtein[b+10])/6
                                                      elif b == 6 : extendedwindow= (NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9]+NewScore10_byProtein[b+10])/7
                                                      elif b == 7 : extendedwindow= (NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9]+NewScore10_byProtein[b+10])/8
                                                      elif b == 8 : extendedwindow= (NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9]+NewScore10_byProtein[b+10])/9
                                                      elif len(NewScore10_byProtein)-b < 7 : extendedwindow= (NewScore10_byProtein[b-10]+NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6])/5
                                                      elif len(NewScore10_byProtein)-b == 7 : extendedwindow= (NewScore10_byProtein[b-10]+NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6])/6
                                                      elif len(NewScore10_byProtein)-b == 8 : extendedwindow= (NewScore10_byProtein[b-10]+NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7])/7
                                                      elif len(NewScore10_byProtein)-b == 9 : extendedwindow= (NewScore10_byProtein[b-10]+NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8])/8
                                                      elif len(NewScore10_byProtein)-b == 10 : extendedwindow= (NewScore10_byProtein[b-10]+NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9])/9 
                                                      else: extendedwindow= (NewScore10_byProtein[b-10]+NewScore10_byProtein[b-9]+NewScore10_byProtein[b-8]+NewScore10_byProtein[b-7]+NewScore10_byProtein[b-6]+NewScore10_byProtein[b+6]+NewScore10_byProtein[b+7]+NewScore10_byProtein[b+8]+NewScore10_byProtein[b+9]+NewScore10_byProtein[b+10])/10
                                                      New_Binary1WindowDiff.append(extendedwindow)


# # Predictor Independent Featuers

# In[51]:


Terminal_Distance= [] 
b = 0
terminaldistance=0
for b in range(0,len(Spot_PredictionScore_byProteins),1):
                                                      if b <len(Spot_PredictionScore_byProteins)/2 : terminaldistance= b/len(Spot_PredictionScore_byProteins)
                                                      else : terminaldistance = ((len(Spot_PredictionScore_byProteins))-(b+1))/len(Spot_PredictionScore_byProteins) 
                                                      Terminal_Distance.append(terminaldistance)


# In[52]:


Terminal_Distance10= [] 
terminaldistance10=0
b = 0
for b in range(0, (len(Spot_PredictionScore_byProteins)), 1):
                                                      if b <10 : terminaldistance10= b
                                                      elif (len(Spot_PredictionScore_byProteins))-(b+1)<10 :terminaldistance10 = ((len(Spot_PredictionScore_byProteins))-(b+1))
                                                      else : terminaldistance10 = 10
                                                      Terminal_Distance10.append(terminaldistance10)


# In[53]:


len(Spot_PredictionScore)


# In[54]:


TestSet_Featuers=pd.DataFrame({'Spot_PredictionScore':Spot_PredictionScore,                
'Spot_BinaryPredictions':Spot_BinaryPredictions,
'Spot_Prediction_ScoreWindow':Spot_Prediction_ScoreWindow,
'Spot_Prediction_ScoreWindowWeighted':Spot_Prediction_ScoreWindowWeighted,
'Spot_Prediction_ScoreWindowDiff':Spot_Prediction_ScoreWindowDiff,                                  
'Spot_Binary_PredictionWindow':Spot_Binary_PredictionWindow,
'Spot_Binary_PredictionWindowWeighted':Spot_Binary_PredictionWindowWeighted,
'Spot_Binary_PredictionWindowDiff':Spot_Binary_PredictionWindowDiff,                                   
'Spot_Prediction_Score_Up':Spot_Prediction_Score_Up,
'Spot_Prediction_Score_Down':Spot_Prediction_Score_Down,
'Spot_Prediction_Score_3AVG':Spot_Prediction_Score_3AVG,
'Spot_Binary_Prediction_Up':Spot_Binary_Prediction_Up, 
'Spot_Binary_Prediction_Down':Spot_Binary_Prediction_Down,
'Spot_Binary_Prediction_3AVG':Spot_Binary_Prediction_3AVG,
'Spot_Disorder_DepthScore':Spot_Disorder_DepthScore,
'Spot_Order_DepthScore':Spot_Order_DepthScore,
'IUPL_PredictionScore':IUPL_PredictionScore,
'IUPL_BinaryPredictions':IUPL_BinaryPredictions,
'IUPL_Prediction_ScoreWindow':IUPL_Prediction_ScoreWindow,
'IUPL_Prediction_ScoreWindowWeighted':IUPL_Prediction_ScoreWindowWeighted,
'IUPL_Prediction_ScoreWindowDiff':IUPL_Prediction_ScoreWindowDiff,                                   
'IUPL_Binary_PredictionWindow':IUPL_Binary_PredictionWindow,
'IUPL_Binary_PredictionWindowWeighted':IUPL_Binary_PredictionWindowWeighted,
'IUPL_Binary_PredictionWindowDiff':IUPL_Binary_PredictionWindowDiff,                                  
'IUPL_Prediction_Score_Up':IUPL_Prediction_Score_Up,
'IUPL_Prediction_Score_Down':IUPL_Prediction_Score_Down,
'IUPL_Prediction_Score_3AVG':IUPL_Prediction_Score_3AVG,
'IUPL_Binary_Prediction_Up':IUPL_Binary_Prediction_Up, 
'IUPL_Binary_Prediction_Down':IUPL_Binary_Prediction_Down,
'IUPL_Binary_Prediction_3AVG':IUPL_Binary_Prediction_3AVG,
'IUPL_Disorder_DepthScore':IUPL_Disorder_DepthScore,
'IUPL_Order_DepthScore':IUPL_Order_DepthScore,
'Consensus_Score':Consensus_Score,
'Consensus_ScoreWindow':Consensus_ScoreWindow,
'Consensus_ScoreWindowWeighted':Consensus_ScoreWindowWeighted,
'Consensus_ScoreWindowDiff':Consensus_ScoreWindowDiff,                                                                     
'New_Score1':New_Score1,
'New_Score1Window':New_Score1Window,
'New_Score1WindowWeighted':New_Score1WindowWeighted,
'New_Score1WindowDiff':New_Score1WindowDiff,                                     
'New_Score2':New_Score2,
'New_Score2Window':New_Score2Window,
'New_Score2WindowWeighted':New_Score2WindowWeighted,
'New_Score2WindowDiff':New_Score2WindowDiff,                                   
'New_Score3':New_Score3,
'New_Score3Window':New_Score3Window,
'New_Score3WindowWeighted':New_Score3WindowWeighted,
'New_Score3WindowDiff':New_Score3WindowDiff,                                                                       
'New_Binary1':New_Binary1,
'New_Binary1Window':New_Binary1Window,
'New_Binary1WindowWeighted':New_Binary1WindowWeighted,
'New_Binary1WindowDiff':New_Binary1WindowDiff,                                                                   
'Terminal_Distance':Terminal_Distance,
'Terminal_Distance10':Terminal_Distance10})
TestSet_Featuers.head()                                


# In[55]:


X_Initial = TestSet_Featuers.iloc[:,0:53]
filename = 'Consensus_Modeltt.sav'
Classifier_Overall = pickle.load(open(filename, 'rb'))
dall = xgb.DMatrix(X_Initial)
Con_Score9= Classifier_Overall.predict(dall)


# In[ ]:


CONSENSUS=""  #Take a string and store all floats
for i in range(len(Con_Score9)):
	CONSENSUS +=str("{:.3f}".format(Con_Score9[i]))+","
CONSENSUS = CONSENSUS[:-1] # To remove the last comma 
print(CONSENSUS)







