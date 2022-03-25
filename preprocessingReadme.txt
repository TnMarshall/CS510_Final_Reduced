To proccess the the data from the original file "ibl_session_mice_data.csv", first run CS510cropDataFromOriginal.m.
This will produce "final510IBLdataReduced.csv". To randomize and zscore the data, run fasterProcessing.m . The original
file for this processing is CS510dataPreprocessingToNumerical.m . However, DO NOT use that file. It uses a for loop that is
much slower and is only kept as a reminder of the issue. The processed data goes into the processedData folder.