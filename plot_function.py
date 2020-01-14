import matplotlib.pyplot as plt
import numpy as np

def plot_freq_domain(freq, amplitude, name):
    """ Shows a lineplot of the given signal.

    Args:
        freq (numpy.ndarray): A list with frequencies on the x axis.
        amplitude (numpy.ndarray): A list with amplitudes on the y axis.
        name(str): A string which is displayed as the plot title.
    """
    plt.figure(figsize=(20,5))
    plt.plot(freq, amplitude)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('amplitude')
    plt.xlim(0, 1400)
    plt.xticks(np.arange(0, 1500, 100))
    plt.grid(which='both')
    plt.title(name)
    plt.show()

    
def plot_hist(all_pitches, ref_min, ref_max, width, height):
    """ Shows a distribution of the pitches in all files used.

    Args:
        all_pitches (list): A list which contains all pitches which are used.
        ref_min (int): The minimum pitch value of the reference note.
        ref_max (int): The maximum pitch value of the reference note.
        width (int): Width of the plot.
        height (int): Height of the plot.
                                  
    Returns:
        A histogramm of the pitch distribution of all files.
    """
         
    fig, ax = plt.subplots(figsize = (width, height)) 
    bins = np.arange(min(all_pitches), max(all_pitches)+1)
    plt.xlim([min(all_pitches)-2, max(all_pitches)+2])

    min_pitch = min(all_pitches)   
    max_pitch = max(all_pitches)

    ref_min = ref_min
    ref_max = ref_max

    plt.hist(all_pitches, bins = bins, range=[min_pitch,max_pitch], alpha=0.5, width=0.8)
    plt.axvline(linewidth=2, color='#fc8803', x=ref_min, label='lowest pitch in reference dataset')
    plt.axvline(linewidth=2, color='r', x=ref_max, label='highest pitch in reference dataset')
    plt.title('pitch distribution')
    plt.xlabel('pitch')
    plt.ylabel('count')
    plt.legend(loc = 'upper left')
    plt.xticks(np.arange(min_pitch, max_pitch, 5))
    plt.close(fig)
    return fig


def plot_bar(df, tone_index, width, height, threshold):
    """ Shows a bar plot of the detected pitches.

    Args:
        df (dataframe): A dataframe which contains all the test data.
        tone_index(int): A index of a tone which is to be analysed.
        width (int): Width of the plot.
        height (int): Height of the plot.
        threshold (float): A value which define if a tone is a 1 or a 0.
                         
    Returns:
        A bar plot of the detected pitches.
    """

    pitch_list = np.arange(40, 77, 1)
    
    fig, ax = plt.subplots(figsize = (width, height))
    index = len(pitch_list)-1

    bar_width = 0.3
    opacity = 0.9

    target_rects = plt.bar(pitch_list, df.target_vec[tone_index], 
                           bar_width, alpha = opacity, label='target_vec', color='#3266a8')

    thresh_rects = plt.bar(pitch_list + 2*bar_width, df.thresholded_vec[tone_index], 
                           bar_width, alpha = opacity, label='thresh_vec', color='#00b82b')

    pred_rects = plt.bar(pitch_list + bar_width, df.norm_pred_vec[tone_index], 
                         bar_width, alpha = opacity, label='norm_pred_vec',color='orange')


    plt.axhline(linewidth=1, color='r', y=threshold, label='threshold')

    plt.xticks(pitch_list + bar_width, pitch_list)
    plt.xlim(pitch_list[0]-0.5, pitch_list[index]+1)
    ax.set_title('pitch detection')
    ax.set_ylabel('intensity')
    ax.set_xlabel('pitch')
    plt.tight_layout()
    plt.title('index from test dataset: ' + str(tone_index))
    plt.legend()
    plt.close(fig)
    return fig


def plot_box(mono_score, poly_score, width, height):
    """ Shows a distribution of the pitch score.

    Args:
        mono_score (pandas.core.series.Series): A list of all calculated scores for each monophonics.
        poly_score (pandas.core.series.Series): A list of all calculated scores for each polyphonics.
        width (int): Width of the plot.
        height (int): Height of the plot.
                                  
    Returns:
        A distribution of the pitch scores as a boxplot.
    """
    pitch_score_data = [mono_score, poly_score]
    fig, ax = plt.subplots(figsize = (width, height))
    # Set the axes ranges and axes labels
    ax.set_xticklabels(['monophonic', 'polyphonic'])
    ax.set_ylabel('pitch-score')
    top = 10
    bottom = -1
    ax.set_ylim(bottom, top)

    ax.set_title('distribution of pitch scores')
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax.boxplot(pitch_score_data)
    ax.text(0.95, -0.6, round(mono_score.median(), 2))
    ax.text(1.95, -0.6, round(poly_score.median(), 2))
    plt.close(fig)
    return fig


def plot_scatter(pitch_pred, pitch_target, width, height):
    """ Shows a scatterplot of the mono pitch detection.

    Args:
        pitch_pred (list): A list with all predicted pitches.
        pitch_target (list): A list with all target pitches.
        width (int): Width of the plot.
        height (int): Height of the plot. 
        
    Returns:
        A scatterplot of the detected mono pitches.
    """

    fig, ax = plt.subplots(figsize = (width, height))
    ax.scatter(pitch_pred, pitch_target, color='#00b82b', s=10)
    ax.set_xlabel('pitch pred')
    ax.set_ylabel('pitch target')
    ax.set_title('monophonic detection')
    plt.xticks(np.arange(40, 96, 5))
    plt.yticks(np.arange(40, 96, 5))
    plt.grid(which='major')
    
    # Create specify x-Labels
    labels = ax.get_xticks().tolist()
    labels[len(labels)-1] = 'x'
    ax.set_xticklabels(labels)
    
    plt.close(fig)
    return fig