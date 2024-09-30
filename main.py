import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from utils.contants import UCR_list
from utils.transfer_learning import pre_train_multi_source, fine_tuning


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a Time Series Transfer Learning with Multi-Source Pre-Training')

    parser.add_argument('--gpus', type=str, default="", help="Sets CUDA_VISIBLE_DEVICES")

    parser.add_argument('--save-pre-trained-model',"-sv-pm", default=False, action="store_true", help="save pre-trainied model to disk?")

    parser.add_argument('--save-fine-tuned-model', "-sv-fm", default=False, action="store_true", help="save fine-tuned model to disk?")

    parser.add_argument("--target", "-t", type=str, help="target dataset to train, if you want to train all the datasets in UCR Archive, input -t experiment")

    parser.add_argument('--model', type=str, default="vgg", help="Set model architecture to train, default = vgg")

    parser.add_argument("--pre-iteration", "-pi", default=10000, type=int, help='itrations for pre-training, default set to 10000')

    parser.add_argument("--transfer-iteration", "-ti", default=5000, type=int, help = "iterations for fine-tuning")

    parser.add_argument("--dataset-number", "-dn", type=int, help="number of datasets for multi-source pre-training")
    
    parser.add_argument("--metric", default="Minimum_Shapelet",type=str, help="metric to rank datasets for source selection")

    parser.add_argument("--fine-tuning-only", default = False, action="store_true", help="Conduct fine-tuning only from saved pre-trained model")

    args = parser.parse_args()

    # Set GPU availability
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus


    def multi_source_transferring(target = args.target, save_path = None):

        if args.fine_tuning_only:
            model_path = "model_save/%s/%s/%s.h5" % (args.metric, args.dataset_number, target)
            if os.path.isfile(model_path):
                print(os.stat(model_path))
                pre_model = keras.models.load_model(model_path)
                loss, acc = fine_tuning(pre_trained_model=pre_model, target=target, nb_iterations=args.transfer_iteration)
                return loss, acc, 0
            else:
                print(f"No pre-trained model found at {model_path}")
                return None, None, 0
        
        else:    
            # source selection
            score_table = pd.read_csv("score/%s.csv" %args.metric, index_col=0)
            score_table = score_table.values
            score_table = score_table.astype(np.float64)
            score_table = score_table.T

            sorted_metric_index = np.argsort(score_table[UCR_list.index(target)])

            ranked_metric_result = sorted_metric_index.astype(int)[:args.dataset_number + 1]

            source_list = np.array(UCR_list)[ranked_metric_result]
            
            if target in source_list:  # delete itself if it is in the source_list
                source_list = np.delete(source_list, np.where(source_list == target))
            else:   # delete the smallest last one
                source_list = np.delete(source_list, -1)
            print("pre-training with multi-sources of :", source_list)

            if not os.path.exists("model_save/%s/%s" %(args.metric, args.dataset_number)):
                    os.makedirs("model_save/%s/%s" %(args.metric, args.dataset_number))

            save_path = "model_save/%s/%s/%s.h5" %(args.metric, args.dataset_number, target)
            
            pre_training_model = pre_train_multi_source(source_list = source_list, target = target, model_architecture = args.model,
                                                        dataset_balancing = True, save_model = args.save_pre_trained_model, 
                                                        metric=args.metric, nb_iterations=args.pre_iteration, save_path = save_path)
            
            loss, acc = fine_tuning(pre_trained_model = pre_training_model, target = target)
            return loss, acc, source_list


    if args.target =="experiment":
        loss_list = []
        acc_list = []
        source_list_list=[]

        for target in UCR_list:
            loss, acc, source_list = multi_source_transferring(target=target)
            loss_list.append(loss)
            acc_list.append(acc)
            source_list_list.append(source_list)

        df_result = pd.DataFrame({
            "dataset": UCR_list,
            "source_list": source_list_list,
            "acc": acc_list,
            "loss": loss_list
        }, index=range(len(UCR_list)))

        df_result.to_csv("result/%s_%s_result.csv" 
                         % (args.metric, args.dataset_number))
        
    else:
        loss, acc, _ = multi_source_transferring()

        print("transfer learning finished")
        print("loss : %s, accuracy : %s" %(loss, acc))
