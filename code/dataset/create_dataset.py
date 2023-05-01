import os
import numpy as np
import pandas as pd
import neurokit2 as nk
import warnings

warnings.filterwarnings("ignore")


# noinspection PyTypeChecker
def process_ecg(signal: pd.Series) -> pd.DataFrame:
    # Clean the signal, then extract peaks and rate
    cleaned_signal = nk.ecg_clean(signal, sampling_rate=1000, method='pantompkins1985')
    peaks, _ = nk.ecg_peaks(cleaned_signal, sampling_rate=1000, method='pantompkins1985', correct_artifacts=True)
    rate = nk.ecg_rate(peaks, sampling_rate=1000, desired_length=len(cleaned_signal))

    # Calculate mean ECG rate and initialize output
    output = pd.DataFrame(data={'HRV_ECG_Rate_Mean': np.mean(rate)}, index=[0])

    # Calculate HRV related features and add to output
    time = nk.hrv_time(peaks, sampling_rate=1000)
    freq = nk.hrv_frequency(peaks, sampling_rate=1000)

    output = pd.concat([output, time, freq], axis=1)

    return output


def process_eda(signal: pd.Series) -> pd.DataFrame:
    # Process the raw signal
    processed, info = nk.eda_process(signal, sampling_rate=1000, method='neurokit')

    # Extract features and add to output frame
    analyzed = nk.eda_intervalrelated(processed)
    analyzed['EDA_Tonic_SD'] = np.nanstd(processed['EDA_Tonic'].values)
    symp = nk.eda_sympathetic(processed['EDA_Clean'], sampling_rate=1000, method='posada')
    analyzed['EDA_Symp'] = symp['EDA_Symp']
    analyzed['EDA_SympN'] = symp['EDA_SympN']
    analyzed['EDA_Autocorrelation'] = nk.eda_autocor(processed['EDA_Clean'], sampling_rate=1000)

    return analyzed


data = pd.DataFrame()

try:
    sub_num = 1
    # go through every subject
    for file in os.listdir('BIRAFFE2/procedure'):
        # extract subject ID
        sub = file[:6]
        print(f'Processing data for {sub} ({sub_num}/117)')
        sub_num += 1

        dtypes_biosigs = {'TIMESTAMP': np.float64, 'ECG': np.float64, 'EDA': np.float64}
        dtypes_procedure = {'TIMESTAMP': np.float64, 'ID': str, 'COND': str,
                            'IADS-ID': str, 'IAPS-ID': str, 'ANS-VALENCE': np.float64,
                            'ANS-AROUSAL': np.float64, 'ANS-TIME': str, 'EVENT': str}

        # read data from BioSigs and Procedure
        biosigs = pd.read_csv(f'BIRAFFE2/biosigs/{sub}-BioSigs.csv', dtype=dtypes_biosigs)
        procedure = pd.read_csv(f'BIRAFFE2/procedure/{sub}-Procedure.csv', sep=';', dtype=dtypes_procedure)

        # delete useless and training entries
        procedure = procedure[procedure['COND'].notnull()]
        procedure = procedure[procedure['COND'] != 'train']

        # go through every image and sound combination
        cases = procedure.shape[0]
        for c in range(cases - 1):
            # divide data into individual combinations
            start = procedure.iloc[c]['TIMESTAMP']
            end = procedure.iloc[c + 1]['TIMESTAMP']

            # detect break between parts
            if end - start > 19:
                continue

            try:
                ecg_signal = biosigs[biosigs['TIMESTAMP'].between(start, end)]['ECG']
                ecg_processed = process_ecg(ecg_signal)

                eda_signal = biosigs[biosigs['TIMESTAMP'].between(start, end)]['EDA']
                eda_processed = process_eda(eda_signal)

            except (ValueError, ZeroDivisionError, IndexError, TypeError, AttributeError):
                if c < 2:
                    print(f'{sub} biosignals data is incomplete. Skipping...')
                    break
                print(f'Error in case {c} condition: {procedure.iloc[c]["COND"]} starting at {start}')
                if biosigs[biosigs['TIMESTAMP'].between(start, end)].shape[0] == 0:
                    print('No data points found in a given time window')
                continue

            # add features to predict
            pred = {'COND:': procedure.iloc[c]['COND'],
                    'ANS_VALENCE': procedure.iloc[c]['ANS-VALENCE'],
                    'ANS_AROUSAL': procedure.iloc[c]['ANS-AROUSAL']}
            pred_frame = pd.DataFrame(data=pred, index=[0])

            # append features to final DataFrame
            features = pd.concat([ecg_processed, eda_processed, pred_frame], axis=1)
            data = pd.concat([data, features])

    # delete NaN values (some columns are always NaN)
    # data.dropna(inplace=True, axis=1)

    # save to file
    print('Writing data to combined_dataset.csv')
    data.to_csv('combined_dataset.csv', index=False)

except FileNotFoundError:
    print('\n    ------------------------ ERROR ------------------------')
    print('''    This script requires two directories in ./BIRAFFE2:
    biosigs - containing biosignals csv files
    procedure - containing procedure csv files''')
