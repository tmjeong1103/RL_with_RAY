from matplotlib import pyplot as plt
import pickle
from config import *

def save_plot_data(plot_dict,file_name):
    # save dictionary to json
    with open('plot_data/%s.json'%file_name,'wb') as fp:
        pickle.dump(plot_dict, fp)
    return print("plot_data saved!")

def draw_graph(file_name):
    # load json file
    with open('plot_data/%s.json'%file_name,'rb') as fp:
        plot_dict = pickle.load(fp)

    # make ep_ret-total step graph
    plt.plot(list(plot_dict.keys()),plot_dict.values(),marker='o')
    plt.xlabel('step')
    plt.ylabel('ep_return')
    plt.title("hdim:%s alpha_pi:[%.4f] alpha_q:[%.4f]\npolyak:[%.4f] gamma:[%.4f] eps:[%.4f] lr:[%.4f]"
              % (hdims, alpha_pi, alpha_q, polyak, gamma, epsilon, lr))
    plt.grid(True, linestyle='--')
    plt.savefig('plot_data/plot_images/%s.png'%file_name,dpi=100)
    #plt.show()