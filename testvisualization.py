from main_hiker import *
import numpy as np
import matplotlib.pyplot as plt
import pickle

######### TEST VISUALIZATION

fnic = "sims/all_AZ0060_t1.pkl"
with open(fnic, 'rb') as f:
    alldata = pickle.load(f)

reps = 10
T = 850
probs = np.loadtxt('test6beh.csv', delimiter=',')
visualization(alldata, reps, T, probs)
plt.show()