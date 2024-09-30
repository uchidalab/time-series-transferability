'''

This code is modified from the following source:
https://stumpy.readthedocs.io/en/latest/Tutorial_Shapelet_Discovery.html

'''

import argparse
from utils.contants import UCR_list
from utils.transfer_learning import load_data
import matplotlib.pyplot as plt
import stumpy
import numpy as np
import utils.datasets as ds
import pandas as pd
import os



def generate_shapelet(dataset, nb_shapelet, len_shapelet):
        
        """
        Generate shapelets according to matrix based shapelet discovery
        :param dataset: name of dataset
        :param nb_shapelet: number of shapelet candidates to discover
        :param len_shapelet: length of shapelet to discover
        """

        # plt.style.use('https://raw.githubusercontent.com/TDAmeritrade/stumpy/main/docs/stumpy.mplstyle')

        # load data
        train_data_file = os.path.join(
            'data', dataset, "%s_TEST.tsv" % dataset)
        x_train, y_train, _, __ = load_data(dataset)

        for i in range(ds.nb_classes(dataset)):
            target_data = x_train[y_train == i]
            target_data = pd.DataFrame(target_data).iloc[:, 1:].reset_index(drop=True)
            opponent_data = x_train[y_train != i]
            opponent_data = pd.DataFrame(opponent_data).iloc[:, 1:].reset_index(drop=True)

            target_data = (target_data.assign(NaN=np.nan)
                    .stack(dropna=False).to_frame().reset_index(drop=True)
                    .rename({0: "Centroid Location"}, axis='columns')
            )
            opponent_data = (opponent_data.assign(NaN=np.nan)
                    .stack(dropna=False).to_frame().reset_index(drop=True)
                    .rename({0: "Centroid Location"}, axis='columns')
            )
            m=len_shapelet
            P_target_target=stumpy.stump(target_data["Centroid Location"], m)[:, 0].astype(float)
            P_target_opponent=stumpy.stump(target_data["Centroid Location"], m, opponent_data["Centroid Location"], ignore_trivial=False)[:, 0].astype(float)

            P_target_target[P_target_target == np.inf] = np.nan
            P_target_opponent[P_target_opponent == np.inf] = np.nan

            P_diff = P_target_target - P_target_opponent

            idx = np.argpartition(np.nan_to_num(P_diff), -nb_shapelet)[-nb_shapelet:]
            shapelets = []
            for j in idx:
                shapelet = target_data.iloc[j : j + m, 0]
                shapelets.append(shapelet)
                # make directory shapelets/len%s_nb%s   %(args.length_shapelet, args.nb_candidate)
            path = "shapelets/len%s_nb%s/%s" %(args.length_shapelet, args.nb_candidate, dataset)
            if not os.path.exists(path):
                os.makedirs(path)
            np.save("%s/%s.npy" %(path, i), shapelets)
        return shapelets

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate shapelets according to matrix based shapelet discovery')

    parser.add_argument("--nb-candidate", type = int, default = 10, help = "number of candidate to discover shapelet from matrix profile")

    parser.add_argument("--length-shapelet", type = int, default = 15, help = "length of shapelet to discover")

    # parser.add_argument("--save-path", type = str, default = None, help = "path to save shapelets")

    args = parser.parse_args()


    for dataset in UCR_list:
        print(dataset, "%s / 128" %UCR_list.index(dataset))
        shapelets = generate_shapelet(dataset, args.nb_candidate, args.length_shapelet)