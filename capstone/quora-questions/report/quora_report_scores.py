import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_scores(df, start_index, filepath_output):
    fig, ax = plt.subplots()
    plt.plot(np.flip(df['Public Score'].values)[start_index:], color='blue', label='Public Score')
    plt.plot(np.flip(df['Private Score'].values)[start_index:], color='red', label='Private Score')
    numbers = np.flip(df['Num'].values)[start_index:]
    explanations = np.flip(df['Source'].values)[start_index:]
    xtick_labels = [str(numbers[i]) + ' (' + explanations[i]  + ')' for i in range(len(numbers))]
    plt.xticks(range(len(xtick_labels)), xtick_labels, rotation=90)
    plt.subplots_adjust(bottom=0.5)
    plt.legend()
    #plt.show()
    fig.set_figheight(10)
    fig.set_figwidth(15)
    plt.savefig(filepath_output)



df = pd.read_csv('data/scores.csv', sep='\s*,\s*')
plot_scores(df, 0, 'scores_all.png')
plot_scores(df, 11, 'scores_latest.png')
