This is the manuscript to produce the entire result of autism wavelet extractor model. <br>
Threre are four main steps to run the script.
1- Download From ABIDE
2- Extract Wavelet
3- Run the model
4- Find best model for each site

1- Downlod the ABIDE image dataset by using the Download_from_abide.py. 
<br>
In this script there are options, that you can use them to download specific files, for example if you want to use cpac (preprocess project's ABIDE preprocess) pipline
change the pipline variable in line 149. More details about the options are in the file. <br>. Downloaded files will placed in the provided path that specified by outdir, 
in the line 176.

2- To Extract wavelet use the Wavelet_Extractor.py. Two important parameters in this file are out_dir and fmri_path. out_dir specified the name of folder, in which the result of extracted_wavelet apeared, the fmri_path is the name of path folder of the downloaded images from ABIDE in the previous step. 

3- After downloading and extrating the feature that are long time consume process, running the model is easy. To run the model all of things that you need to do is the call Classifiy_Wavelet_For_Each_Site function from the Making_Model.py.
The result will come out in the result directory so it is neccessary to create a new direcotry with result's name before running this script. 


4- Finally to find the best model and draw some interset plot for each site, please run the Wavelet_Best.py and Analysis_Result_BoxPlot.py respectivly, the first script find the best model for each site based on the provided result in the third step and the second script draw interest plot from the output of the wavelet_best.py.




