from matplotlib import pyplot as plt
import pickle
from config import *

def save_plot_data(plot_dict,file_name):
    # save dictionary to json
    with open('plot_data/%s.json'%file_name,'wb') as fp:
        pickle.dump(plot_dict, fp)
    return print("plot_data saved!")

def draw_subplt_graph(file_name1, file_name2):
    # load json file
    with open('plot_data/%s.json'%file_name1,'rb') as fp:
        plot_dict1 = pickle.load(fp)
    with open('plot_data/%s.json'%file_name2,'rb') as fp2:
        plot_dict2 = pickle.load(fp2)

    plt.subplots_adjust(hspace=0.5)
    # make ep_ret-total step graph
    plt.subplot(2,1,1)
    plt.plot(list(plot_dict1.keys()),plot_dict1.values(),marker='o')
    plt.xlabel('step')
    plt.ylabel('ep_return')
    plt.title("hdim:%s gamma:[%.4f] lam:[%.4f] clip_ratio:[%.4f]\n epsilon:[%.4f] ep_len_rollout:[%d]"
              %(hdims,gamma,lam,clip_ratio,epsilon,ep_len_rollout))
    plt.grid(True, linestyle='--')

    plt.subplot(2,1,2)
    plt.plot(list(plot_dict2.keys()),plot_dict2.values(),marker='o')
    plt.xlabel('time')
    plt.ylabel('ep_return')
    # plt.title("hdim:%s gamma:[%.4f] lam:[%.4f] clip_ratio:[%.4f]\n epsilon:[%.4f] ep_len_rollout:[%d]"
    #           %(hdims,gamma,lam,clip_ratio,epsilon,ep_len_rollout))
    plt.grid(True, linestyle='--')

    plt.savefig('plot_data/plot_images/%s.png'%file_name1,dpi=100)
    #plt.show()

def draw_graph(file_name):
    # load json file
    with open('plot_data/%s.json'%file_name,'rb') as fp:
        plot_dict = pickle.load(fp)

    # make ep_ret-total step graph
    plt.subplot(2,1,1)
    plt.plot(list(plot_dict.keys()),plot_dict.values(),marker='o')
    plt.xlabel('step')
    plt.ylabel('ep_return')
    plt.title("hdim:%s gamma:[%.4f] lam:[%.4f] clip_ratio:[%.4f]\n epsilon:[%.4f] ep_len_rollout:[%d]"
              %(hdims,gamma,lam,clip_ratio,epsilon,ep_len_rollout))
    plt.grid(True, linestyle='--')

    plt.savefig('plot_data/plot_images/%s.png'%file_name,dpi=100)
    #plt.show()