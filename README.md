# Classifying Categorical Data Based on Adoptive Hamming Distance


## Abstract

Selection of the distance measure is important in the categorical pattern classification problem: the Hamming distance, Value Difference Metric, and Class Dependent Weight Dissimilarity have been widely used distance measure. However, the Hamming Distance does not address the difference between categorical values according to class. Regarding a specific domain values as equivalent may imply an improved classification performance.

This program finds such pairs of categorical values in a specific class to improve classification performance. The adoptive distance is searched by a hill-climbing manner in which the classification accuracy is used as an indicator of the equivalence of categorical values.

This software is a Matlab implementation of AHD method, highy specialized on problems of categorical data set classification. The original version of this program was written by Jae-Sung Lee.

### [Paper]
The main technical ideas behind how this program works appear in these papers:

Jae-Sung Lee, and Dae-Won Kim, [“Classifying Categorical Data Based on Adoptive Hamming Distance,”](https://www.jstage.jst.go.jp/article/transinf/E93.D/1/E93.D_1_189/_article/-char/ja/) IEICE Transactions on Information and Systems E93-D(1):189-192, 2010.

R. Paredes and E. Vidal, [“A class-dependent weighted dissimilarity measure for nearest neighbor classification problem,”](https://www.sciencedirect.com/science/article/pii/S0167865500000647?via%3Dihub) Pattern Recognition 21(12):1027-1036, 2000.


## License

This program is available for download for non-commercial use, licensed under the GNU General Public License, which is allows its use for research purposes or other free software projects but does not allow its incorporation into any type of commerical software.

## Sample Input and Output

It will find distance matrix for the categorical data, ouputting the results to a matrix named for user-specified variable. This code can executed under Matlab command window.

### [Usage]:
   `>> domain_info = dt_count( [TrainData(1:end-1);TestData] );` \
   `>> DT = ddcs( TrainData, domain_info );` \
   `>> distMat = ahd( TrainData, TestData, DT );`

### [Description]
   TrainData – a matrix that is composed of features and answer \
   domain_info – the structure of data set(how many domain values in each feature?) \
   ddcs – finding categorical value pairs those are improving classification performance \
   distMat – distance matrix between Train and Test Pattern based on ddcs


Currently, AHD reads a train data set that is composed of (features+answer). Thus the last column of train data set(last feature) is treated as answer. By convention in AHD input matrix rows represent data (e.g. patterns) and columns represent features and answer for each pattern.

The output matrix contains the distance matrix between train patterns(Row) and test patterns(Column). An example output matrix when three train patterns and two test patterns are inputted might look like this.

         TS1   TS2
    TR1   0   3
    TR2   1   2
    TR3   2   1
