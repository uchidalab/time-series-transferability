import utils.datasets as ds
import numpy as np
import pandas as pd
from utils.constants import UCR_list
import os

length = 15
nb_candidate = 10
nb_len = "len%s_nb%s" %(length, nb_candidate)
print(nb_len)

dataset_name = UCR_list

result_all_min = []
result_all_avg= []
for d in dataset_name:
    # load shapelets
    distance_list_min = []
    distance_list_avg = []
    for d2 in dataset_name:
        distance = np.inf
        distance_avg = 0
        if d2 == d:
            distance = 0
            distance_avg = 0
        else:
            for i in range(ds.nb_classes(d)):
                A = np.load("shapelets/%s/%s/%d.npy" % (nb_len, d, i))
                temp = np.inf
                for k in range(nb_candidate):
                    for j in range(ds.nb_classes(d2)):
                        B = np.load("shapelets/%s/%s/%d.npy" % (nb_len, d2, j))
                        temp_dist=np.linalg.norm(A[k]-B, axis=1)
                        if temp > np.min(temp_dist):
                            temp = np.min(temp_dist)
                        distance_avg += np.sum(temp_dist)
                        if np.min(temp_dist) < distance:
                            distance = np.min(temp_dist)

            distance_avg /= (ds.nb_classes(d) * ds.nb_classes(d2) * (nb_candidate ^ 2))

        print(d, d2, distance, distance_avg)
        distance_list_min.append(distance)
        distance_list_avg.append(distance_avg)
    result_all_min.append(distance_list_min)
    result_all_avg.append(distance_list_avg)


if os.path.exists("score") == False:
    os.makedirs("score")
df = pd.DataFrame(result_all_min, index=dataset_name,columns=dataset_name)
df.to_csv('score/shapelet_minimum_%s.csv' %nb_len)
df_avg = pd.DataFrame(result_all_avg, index=dataset_name, columns=dataset_name)
df_avg.to_csv('score/shapelet_average_%s.csv' %nb_len)
