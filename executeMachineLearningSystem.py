import ast
import glob
import json
import os
from tempfile import TemporaryFile

import numpy
import problog
import time
from problog.cnf_formula import CNF
from problog.ddnnf_formula import DDNNF
from problog.formula import LogicDAG
from sklearn.metrics import mean_squared_error, mean_absolute_error
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from EntitiesOfTweet import *
from createModel import createModel
from evaluationOfModel import evalutionOfTrainedModel
from functions import *
from problog.program import PrologString
from problog.core import ProbLog
from problog import get_evaluatable
from problog.engine import DefaultEngine
from problog.logic import Term
from problog.learning import lfi
from problog.tasks import sample
from problog.tasks.dtproblog import dtproblog
import collections
from collections import Counter
from itertools import chain
import sys
import csv
import pandas as pd
import numpy as np

#delete trained model if exists
if os.path.exists('models/model.txt'):
    path = 'models/*'
    r = glob.glob(path)
    for i in r:
        os.remove(i)
print("first dataset for training")
createModel('train1.csv')
print("second dataset for training")
createModel('train2.csv')
print("third dataset for training")
createModel('train3.csv')
print('start the evaluation process, please wait...')
y_pred=evalutionOfTrainedModel('testSet.csv')
y_true=[]
for i in range(0,len(y_pred)):
    y_true.append(1)

print("The mean squared error of our system is (MSE): ",mean_squared_error(y_true, y_pred))
print("The root mean squared error of our system is (RMSE): ", numpy.sqrt(mean_squared_error(y_true, y_pred)))
print("The mean absolute error of our system is (MAE):",mean_absolute_error(y_true, y_pred))