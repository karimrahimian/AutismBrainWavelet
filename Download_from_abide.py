# download_abide_preproc.py
#
# Author: Daniel Clark, 2015
# Updated to python 3 and to support downloading by DX, Cameron Craddock, 2019

"""
This script downloads data from the Preprocessed Connetomes Project's
ABIDE Preprocessed data release and stores the files in a local
directory; users specify derivative, pipeline, strategy, and optionally
age ranges, sex, site of interest
Usage:
    python download_abide_preproc.py -d <derivative> -p <pipeline>
                                     -s <strategy> -o <out_dir>
                                     [-lt <less_than>] [-gt <greater_than>]
                                     [-x <sex>] [-t <site>]
"""


# Main collect and download function
def collect_and_download(derivative, pipeline, strategy, out_dir, less_than, greater_than, site, sex, diagnosis):
    """
    Function to collect and download images from the ABIDE preprocessed
    directory on FCP-INDI's S3 bucket
    Parameters
    ----------
    derivative : string
        derivative or measure of interest
    pipeline : string
        pipeline used to process data of interest
    strategy : string
        noise removal strategy used to process data of interest
    out_dir : string
        filepath to a local directory to save files to
    less_than : float
        upper age (years) threshold for participants of interest
    greater_than : float
        lower age (years) threshold for participants of interest
    site : string
        acquisition site of interest
    sex : string
        'M' or 'F' to indicate whether to download male or female data
    diagnosis : string
        'asd', 'tdc', or 'both' corresponding to the diagnosis of the
        participants for whom data should be downloaded
    Returns
    -------
    None
        this function does not return a value; it downloads data from
        S3 to a local directory
    :param derivative:
    :param pipeline:
    :param strategy:
    :param out_dir:
    :param less_than:
    :param greater_than:
    :param site:
    :param sex:
    :param diagnosis:
    :return:
    """

    # Import packages
    import os
    import urllib.request as request

    # Init variables
    mean_fd_thresh = 0.2
    s3_prefix = 'https://s3.amazonaws.com/fcp-indi/data/Projects/'\
                'ABIDE_Initiative'
    s3_pheno_path = '/'.join([s3_prefix, 'Phenotypic_V1_0b_preprocessed1.csv'])

    # Format input arguments to be lower case, if not already
    derivative = derivative.lower()
    pipeline = pipeline.lower()
    strategy = strategy.lower()

    # Check derivative for extension
    if 'roi' in derivative:
        extension = '.1D'
    else:
        extension = '.nii.gz'

    # If output path doesn't exist, create it
    if not os.path.exists(out_dir):
        print('Could not find {0}, creating now...'.format(out_dir))
        os.makedirs(out_dir)

    # Load the phenotype file from S3
    s3_pheno_file = request.urlopen(s3_pheno_path)
    pheno_list = s3_pheno_file.readlines()
    print(pheno_list[0])

    # Get header indices
    header = pheno_list[0].decode().split(',')
    try:
        site_idx = header.index('SITE_ID')
        file_idx = header.index('FILE_ID')
        age_idx = header.index('AGE_AT_SCAN')
        sex_idx = header.index('SEX')
        dx_idx = header.index('DX_GROUP')
        mean_fd_idx = header.index('func_mean_fd')
    except Exception as exc:
        err_msg = 'Unable to extract header information from the pheno file: {0}\nHeader should have pheno info:' \
                  ' {1}\nError: {2}'.format(s3_pheno_path, str(header), exc)
        raise Exception(err_msg)

    # Go through pheno file and build download paths
    print('Collecting images of interest...')
    s3_paths = []
    for pheno_row in pheno_list[1:]:

        # Comma separate the row
        cs_row = pheno_row.decode().split(',')

        try:
            # See if it was preprocessed
            row_file_id = cs_row[file_idx]
            # Read in participant info
            row_site = cs_row[site_idx]
            row_age = float(cs_row[age_idx])
            row_sex = cs_row[sex_idx]
            row_dx = cs_row[dx_idx]
            row_mean_fd = float(cs_row[mean_fd_idx])
        except Exception as e:
            err_msg = 'Error extracting info from phenotypic file, skipping...'
            print(err_msg)
            continue
        # If the filename isn't specified, skip
        if row_file_id == 'no_filename':
            continue
        # If mean fd is too large, skip
        if row_mean_fd >= mean_fd_thresh:
            continue
        # Test phenotypic criteria (three if's looks cleaner than one long if)
        # Test sex
        if (sex == 'M' and row_sex != '1') or (sex == 'F' and row_sex != '2'):
            continue
        if (diagnosis == 'asd' and row_dx != '1') or (diagnosis == 'tdc' and row_dx != '2'):
            continue
        # Test site
        if site is not None and site.lower() != row_site.lower():
            continue
        # Test age range
        if greater_than < row_age < less_than:
            filename = row_file_id + '_' + derivative + extension
            s3_path = '/'.join([s3_prefix, 'Outputs', pipeline, strategy, derivative, filename])
            print('Adding {0} to download queue...'.format(s3_path))
            s3_paths.append(s3_path)
        else:
            continue

    # And download the items
    total_num_files = len(s3_paths)
    for path_idx, s3_path in enumerate(s3_paths):
        rel_path = s3_path.lstrip(s3_prefix)
        download_file = os.path.join(out_dir, rel_path)
        download_dir = os.path.dirname(download_file)
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
        try:
            if not os.path.exists(download_file):
                print('Retrieving: {0}'.format(download_file))
                request.urlretrieve(s3_path, download_file)
                print('{0:3f}% percent complete'.format(100*(float(path_idx+1)/total_num_files)))
            else:
                print('File {0} already exists, skipping...'.format(download_file))
        except Exception as exc:
            print('There was a problem downloading {0}.\n Check input arguments and try again.'.format(s3_path))

    # Print all done
    print('Done!')

derivative = 'func_preproc'
pipline = 'cpac'
strategy = 'filt_global'
out_dir = 'data/'

less_than = 200
greater_than = -1
site = None
sex = None
diagnosis = 'both'

collect_and_download(derivative,pipline,strategy,out_dir,less_than,greater_than,site,sex,diagnosis)


# web site : https://github.com/preprocessed-connectomes-project/abide

"""
The download_abide_preproc.py script allows any user to download outputs from the ABIDE preprocessed data release. The user specifies the desired derivative, pipeline, and noise removal strategy of interest, and the script finds the data on FCP-INDI's S3 bucket, hosted by Amazon Web Services, and downloads the data to a local directory. The script also allows for phenotypic specifications for targeting only the particpants whose information meets the desired criteria; these specifications include: diagnosis (either ASD, TDC, or both), an age range (e.g. particpants between 2 and 30 years of age), sex (male or female), and site (location where the images where acquired from). * Note the script only downloads images where the functional image's mean framewise displacement is less than 0.2.

At a minimum, the script needs a specific derivative, pipeline, and strategy to search for.
Acceptable derivatives include:
- alff (Amplitude of low frequency fluctuations)
- degree_binarize (Degree centrality with binarized weighting)
- degree_weighted (Degree centrality with correlation weighting)
- eigenvector_binarize (Eigenvector centrality with binarized weighting)
- eigenvector_weighted (Eigenvector centrality with correlation weighting)
- falff (Fractional ALFF)
- func_mask (Functional data mask)
- func_mean (Mean preprocessed functional image)
- func_preproc (Preprocessed functional image)
- lfcd (Local functional connectivity density)
- reho (Regional homogeneity)
- rois_aal (Timeseries extracted from the Automated Anatomical Labeling atlas)
- rois_cc200 (" " from Cameron Craddock's 200 ROI parcellation atlas)
- rois_cc400 (" " " 400 ROI parcellation atlas)
- rois_dosenbach160 (" " from the Dosenbach160 atlas)
- rois_ez (" " from the Eickhoff-Zilles atlas)
- rois_ho (" " from the Harvard-Oxford atlas)
- rois_tt (" " from the Talaraich and Tournoux atlas)
- vmhc (Voxel-mirrored homotopic connectivity)

Acceptable pipelines include:
- ccs
- cpac
- dparsf
- niak

Acceptable strategies include:
- filt_global (band-pass filtering and global signal regression)
- filt_noglobal (band-pass filtering only)
- nofilt_global (global signal regression only)
- nofilt_noglobal (neither)

For example, to download all particpants across all sites' ReHo images processed using C-PAC, without any frequency filtering or global signal regression:
    python download_abide_preproc.py -d reho -p cpac -s nofilt_noglobal -o /path/to/local/download/dir

The script will then search for and download the data to the local directory specified with the -o flag.

Participants can also be selected based on phenotypic information. For example, to download the same outputs from the previous example, but using only male ASD participants scanned from Caltech between the ages of 2 and 30 years:
    python download_abide_preproc.py -a -d reho -p cpac -s nofilt_noglobal -o /path/to/local/download/dir -gt 2 -lt 30 -x M -t Caltech

For more information on the ABIDE preprocessed initiative, please check out http://preprocessed-connectomes-project.github.io/abide
"""