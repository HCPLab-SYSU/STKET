import os, sys
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches


result_num = 3

def read_file(file_path):

    data = np.loadtxt(file_path, delimiter=',')
    splits = ['stable', 'low', 'medium', 'high']

    result = {}
    for idx, split in enumerate(splits):
        _data = data[idx]

        result[split] = {'PredCls': [np.arange(result_num), _data[:result_num]],
                         'SGCls':   [np.arange(result_num), _data[result_num:-result_num]],
                         'SGGen':   [np.arange(result_num), _data[-result_num:]],}

    return result

def draw_figure(data_path, task_name):

    color_dict = {'STTran': 'b', 'Ours': 'r'}
    line_dict = {'stable': '-', 'low': '--', 'medium': '-.', 'high': ':'}

    for file_name in os.listdir(data_path):
        if file_name[-3:] != 'txt':
            continue
        assert file_name[:-4] in color_dict, print(file_name)

        draw_data = read_file(os.path.join(data_path, file_name))

        for split_name, results in draw_data.items():
            result = results[task_name]
            plt.plot(result[0], result[1], linestyle=line_dict[split_name], color=color_dict[file_name[:-4]])\

    plt.ylabel('Recall')
    plt.xticks(np.arange(result_num), ['R@10', 'R@20', 'R@50'])

    plt.legend(handles=[mpatches.Patch(color=color, label=method) for method, color in color_dict.items()], 
                    loc='upper right', 
                    prop = {'size': 8}
                    )
                    #bbox_to_anchor=(1.05,0.85), 
                    #borderaxespad=0)

    plt.legend(handles=[mlines.Line2D(xdata=[0], ydata=[0], linestyle=linestyle, color='black', label=split_name) for split_name, linestyle in line_dict.items()],
               loc='upper right',
               prop = {'size': 7}
               )
    

    #plt.gca().add_artist(l1)

    plt.savefig(fname=os.path.join(data_path, 'result_{}.pdf'.format(task_name)), format="pdf", bbox_inches='tight')


if __name__ == '__main__':
    draw_figure(sys.argv[1], sys.argv[2])
