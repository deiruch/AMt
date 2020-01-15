from scipy.fftpack import fft
import numpy as np
from gurobipy import *
from IPython.display import Audio
import xml.etree.ElementTree as et
import pandas as pd
import soundfile


def fourier(data, rate):
    """ Return the Fast Fourier Transform sample frequencies and amplitudes.

    Args:
        data(numpy.ndarray): A list of data points of the given signal
        rate(int): The number of sample rate

    Returns:
        freq(numpy.ndarray): A list of sample frequencies.
        ampl(numpy.ndarray): A list of sample amplitudes.

    """
    N = len(data)

    T = 1.0 / rate
    x = np.linspace(0.0, N*T, N)

    yf = fft(data)
    freq = np.linspace(0.0, 1.0/(2.0*T), N//2)
    ampl = 2.0/N * np.abs(yf[0:N//2])
    return freq, ampl


def snip_wav(data, rate, start_sec, end_sec):
    """ Cuts down the datapoints.

    Args:
        data(numpy.ndarray): A list of data points of the given signal
        rate(int): The number of sample rate
        start_sec(float): The onset where the signal starts.
        end_sec(float): The offset where the signal ends.

    Returns:
        snip_data(numpy.ndarray): A binarised vector.
        rate(int): The number of sample rate
        
    """
    if start_sec > 0:
        start_point = int((rate * start_sec)-1)
    else:
        start_point = 0
        
    end_point = int((rate * end_sec)-1)
    snip_data = data[start_point:end_point]
    return snip_data, rate


def calc_target_vec(pitch):
    """ Returns a binarised vector, where 1 indicates the existence of the pitch, and 0 if not.

    Args:
        pitch(numpy.ndarray): The pitch which is to be binarised.

    Returns:
        target_vec(numpy.ndarray): The a binarised vector.

    """
    pitch_start = 40
    pitch_end = 76
    combi_vec = []
    for element in pitch:
        pitch_vec = []
        if pitch_start <= element <= pitch_end:
            for pitch_actual in range(pitch_start, pitch_end+1):
                if pitch_actual == element:
                    pitch_vec.append(1)
                else:
                    pitch_vec.append(0)
            combi_vec.append(pitch_vec)
        else:
            print('Pitch {} does not lie within the range({} - {})!'.format(pitch, pitch_start, pitch_end))
            return[0] * (pitch_end - pitch_start + 1) # for harmonics
    target_vec = [sum(x) for x in zip(*combi_vec)]
    return np.asarray(target_vec)


def norm_vec(vector):
    """ Returns a normalised vector, where the sum of its square equals 1.

    Args:
        vector (numpy.ndarray): The vector which is to be normalised.

    Returns:
        norm_v(numpy.ndarray): The vector which is normalised.
        vector(numpy.ndarray): The vector which cannot be normalised.

    """
    norm_factor = np.linalg.norm(vector)
    if (norm_factor != 0):
        norm_v = vector/norm_factor
        return np.asarray(norm_v)
    else:
        print('The vector cannot be normalised.')
        return np.asarray(vector)

    
def metric(target_v, approx_v):
    """ Returns the sum of differences between two vectors.

    Args:
        target_v (numpy.ndarray):The vector which is to be approximated.
        approx_v (numpy.ndarray):The actual approximated vector calculated from Gurobi.


    Returns:
        diff_sum(numpy.float64):The sum of differences which is normalised.   

    """
    norm_v = norm_vec(approx_v)
    diff = target_v - norm_v
    diff_sum = np.sum(np.abs(diff))
    return diff_sum  


def vec_to_pitch(vec):
    """ Converts a binary vector to its pitches.

    Args:
        vec(ndarray): The binary vector to be converted.

    Returns:
        pitch(ndarray): A vector with pitches.

    """
    pitch_actual = 40
    pitch = []
   
    for elem in vec:
        if elem == 1:
            pitch.append(pitch_actual)
        pitch_actual += 1
    return pitch


def read_xml_to_df(path, df_cols, offset_sec, duration_sec, num_data_points):
    """ Converts an xml file to a dataframe.

    Args:
        path(str): The path where the xml is located
        df_cols(list): A list of strings which the dataframe should contain of. 
        offset_sec(float): The onset where the signal starts. 
        duration_sec(float): The duration of the signal
        num_data_points(int): Defines the number of data points

    Returns:
        df(data frame): The converted xml file as a dataframe.

    """
    dataset = re.search(r'dataset.*\b', path).group(0)
    path_xml = path + "annotation"
    path_wav = path + "audio"  
    df = pd.DataFrame(columns=df_cols)
    
    for xml_file in sorted(glob.glob(os.path.join(path_xml, '*.xml'))):
        tree = et.parse(xml_file)
        root = tree.getroot()
        all_events = []
       
        for globalParam in root.findall('globalParameter'):
            audio_name = globalParam.find('audioFileName').text
            audio_name = audio_name.replace("\\", "")
            
            wav_file = path_wav + '/' + audio_name
            data, rate = soundfile.read(wav_file)

        for transcription in root.findall('transcription'):

            for event in transcription.findall('event'):
                event_data = []
                event_data.append(dataset)
                event_data.append(audio_name)

                for elem in df_cols[len(event_data):]:
                    if event is not None:
                        if event.find(elem) is not None:
                            event_data.append(event.find(elem).text)
                            
                        elif elem == df_cols[3]:
                            onset_sec = event.find('onsetSec').text
                            event_data.append(onset_sec)
                            start_sec = round(offset_sec + float(onset_sec), 3)
                            end_sec = round(start_sec + duration_sec, 3)

                            data_snip, rate_snip = snip_wav(data, rate, start_sec, end_sec)
                            freq, amplitude = fourier(data_snip, rate_snip)
                                                       
                        elif elem == 'amplitude':
                            event_data.append(norm_vec(amplitude[:num_data_points]))
                            #event_data.append(amplitude[:num_data_points]*amplitude[:num_data_points])


                        elif elem == 'frequency':
                            event_data.append(freq[:num_data_points])
                        else:
                            event_data.append(None)
                    else:
                        event_data.append(None)

                all_events.append({df_cols[i]: event_data[i] for i, _ in enumerate(df_cols)})
                
        combi_events = mono_poly_detection(all_events)
        df = df.append(pd.DataFrame(combi_events, columns=df_cols), ignore_index=True)
        
    return df


def mono_poly_detection(events):
    """ Detects whether a pitch is played alone or as a chord.

    Args:
        events(numpy.ndarray): A list where all events of a file 

    Returns:
        combi_events(numpy.ndarray): A list where all pitches played within 70ms is stored.

    """
    poly_time = 0.07
    
    dataset = events[0]['dataset']
    audioFileName = events[0]['audio_file_name']
    ampl = events[0]['amplitude']
    freq = events[0]['frequency']
    
    onsets = []
    pitches = []
    combi_events = []
    
    for event in events:
        onsets.append(float(event['onset_sec']))
        pitches.append(int(event['pitch']))
        
    pitch_array = [pitches[0]]
    new_onset = onsets[0]
    
    
    for i in range(len(events)-1):
        if (onsets[i + 1]- onsets[i]) <= poly_time:
            pitch_array.append(pitches[i+1])
            new_onset = onsets[i+1]
            ampl = events[i+1]['amplitude']
            freq = events[i+1]['frequency']
            
        else:
            combi_events.append({'dataset': dataset,
                               'audio_file_name': audioFileName,
                               'pitch': pitch_array,
                               'onset_sec': new_onset,
                               'amplitude': ampl,
                               'frequency': freq,
                              })
            
            pitch_array = [pitches[i+1]]
            new_onset = onsets[i+1]
            ampl = events[i+1]['amplitude']
            freq = events[i+1]['frequency']
            
    combi_events.append({'dataset': dataset,
                   'audio_file_name': audioFileName,
                   'pitch': pitch_array,
                   'onset_sec': new_onset,
                   'amplitude': ampl,
                   'frequency': freq,
                  })

    return combi_events