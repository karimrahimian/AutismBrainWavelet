import pywt
import pywt.data
import numpy as np
from scipy.stats import entropy
from scipy.stats import linregress
from nilearn import image
import os
import glob
import pandas
out_dir = "Wavelet_Extractor/Part5"

filter_bank = ['db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'bior1.3', 'bior1.5', 'bior2.2', 'bior2.4',
                   'bior2.6', 'bior3.1', 'bior3.3', 'bior3.5', 'bior3.7' ,
                   'bior4.4', 'bior5.5', 'coif1', 'coif2', 'coif3', 'coif4', 'coif5', 'sym2', 'sym3', 'sym4', 'sym5', 'sym6', 'sym7', 'sym8']

def creating_signal_vector(twoD_features_extracted,filter):  # The function which will give us the WaveletDifumo256 feature vectors.

    coeffs = pywt.dwt2(twoD_features_extracted, filter)
    # LL, (LH, HL, HH),(LH1,HL1,HH1),(LH2, HL2, HH2),(LH3, HL3, HH3) = coeffs
    LL, ( LH, HL, HH) = coeffs  # separate the Approximate coefficient and detail coefficient to 4 signals

    # generating the criteria including max, min, median, average, Standard deviation,energy, entopy,
    # slope and SED error for details coeeficient towards approximate coefficient
    # overal energy and wavelength for overal signal and write in the file for each filter bank
    LLenergy = np.sqrt(np.sum(np.array(LL[-1]) ** 2)) / len(LL[-1])
    LHenergy = np.sqrt(np.sum(np.array(LH[-1]) ** 2)) / len(LH[-1])
    HLenergy = np.sqrt(np.sum(np.array(HL[-1]) ** 2)) / len(HL[-1])
    HHenergy = np.sqrt(np.sum(np.array(HH[-1]) ** 2)) / len(HH[-1])
    overalEnergy = np.sqrt(np.sum(np.array(coeffs[-1]) ** 2)) / len(coeffs[-1])
    LLwavelength = len(LL)

    LL = np.concatenate(LL)
    LH = np.concatenate(LH)
    HL = np.concatenate(HL)
    HH = np.concatenate(HH)

    LL1energy = np.sqrt(np.sum(np.array(LL[-1]) ** 2)) / len(LL)
    LH1energy = np.sqrt(np.sum(np.array(LH[-1]) ** 2)) / len(LH)
    HL1energy = np.sqrt(np.sum(np.array(HL[-1]) ** 2)) / len(HL)
    HH1energy = np.sqrt(np.sum(np.array(HH[-1]) ** 2)) / len(HH)

    LHslope, LHintercept, LHr_value, LHp_value, LHstd_err = linregress(LL, LH)
    HLslope, HLintercept, HLr_value, HLp_value, HLstd_err = linregress(LL, HL)
    HHslope, HHintercept, HHr_value, HHp_value, HHstd_err = linregress(LL, HH)

    maxLL = np.max(LL)
    minLL = np.min(LL)
    AvLL = np.average(LL)
    MedianLL = (minLL + maxLL) / 2

    pd_series = pandas.Series(LL)
    counts = pd_series.value_counts()
    LLentropy = entropy(counts)

    maxLH = np.max(LH)
    minLH = np.min(LH)
    AvLH = np.average(LH)
    MedianLH = (minLH + maxLH) / 2

    pd_series = pandas.Series(LH)
    counts = pd_series.value_counts()
    LHentropy = entropy(counts)

    maxHL = np.max(HL)
    minHL = np.min(HL)
    AvHL = np.average(HL)
    MedianHL = (minHL + maxHL) / 2

    pd_series = pandas.Series(HL)
    counts = pd_series.value_counts()
    HLentropy = entropy(counts)

    maxHH = np.max(HH)
    minHH = np.min(HH)
    AvHH = np.average(HH)
    MedianHH = (minHH + maxHH) / 2

    pd_series = pandas.Series(HH)
    counts = pd_series.value_counts()
    HHentropy = entropy(counts)

    LL = np.std(LL)
    LH = np.std(LH)
    HL = np.std(HL)
    HH = np.std(HH)

    features = [maxLL, maxLH, maxHL , maxHH , minLL , minLH,
                minHL , minHH ,AvLL , AvLH, AvHL , AvHH , MedianLL ,
                MedianLH , MedianHL , MedianHH ,
                LLentropy, LHentropy , HLentropy , HHentropy,
                HLenergy , LLenergy , LHenergy , HHenergy , LHslope ,
                HLslope, HHslope , LHstd_err ,  HLstd_err,
                HHstd_err,   LL1energy , LH1energy, HL1energy, HH1energy,LL , LH , HL , HH]
    return features
def Save_To_File( filename, corrmatrix):
    basename = os.path.basename(filename)
    newname = basename[0:len(basename) - 7]
    np.save("{}/{}/{}.npy".format(out_dir, flt_bank ,newname), corrmatrix)
def Wave_Let_For_Bold_Signal( dirpath):
    for file in glob.glob(dirpath):
        print (file)
        image2d = []
        img = image.load_img(file)
        data = np.asarray(img.dataobj)
        for x in range(data.shape[0]):
            for y in range(data.shape[1]):
                for z in range(data.shape[2]):
                    bold_signal = data[x, y, z, :]
                    image2d.append(bold_signal)
        image2d = np.array(image2d)
        wave_features = creating_signal_vector(image2d,flt_bank)
        Save_To_File(file, np.array(wave_features))
def Wave_Let_For_Bold_Signal_Split_Matrix( dirpath):
    for file in glob.glob(dirpath):
        print(file)
        basename = os.path.basename(file)
        newname = basename[0:len(basename) - 7]
        if (os.path.exists("{}/{}/{}.npy".format(out_dir, flt_bank, newname))):
            continue

        image2d = []
        img = image.load_img(file)
        data = np.asarray(img.dataobj)
        for x in range(data.shape[0]):
            for y in range(data.shape[1]):
                for z in range(data.shape[2]):
                    bold_signal = data[x, y, z, :]
                    image2d.append(bold_signal)

        image2d = np.array(image2d)
        data = image2d[0:271000,:]
        data = np.split(data , 5)
        total_features = []
        for matrix in data:
            wave_features = creating_signal_vector(matrix,flt_bank)
            total_features.append(wave_features)
        last_part = image2d[271000:,:]
        wave_features = creating_signal_vector(last_part, flt_bank)
        total_features.append(wave_features)
        Save_To_File(file, wave_features)

fmri_path = "data/ABIDE_pcp/newdata/*.gz"

for filter in filter_bank:
    print ("**********************************************************{}*****************************************".format(filter))
    flt_bank = filter
    try:
        os.mkdir("{}/{}".format(out_dir, flt_bank))
    except:
        pass
    Wave_Let_For_Bold_Signal_Split_Matrix(fmri_path)
