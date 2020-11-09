#!/bin/sh

# set -x

# "sh /home/leandro/Escritorio/nuevo/AG_Emos_FAU/wekaknn.sh rnk"
# "sh /home/leandro/Escritorio/nuevo/AG_Emos_FAU/wekaknn.sh rnk trn_file tst_file"


trn_file=data/madelon/madelon.trn.arff
tst_file=data/madelon/madelon.tst.arff



# nk=5
nk=15


path=.
here=$path

# cromo=$path/genetico/$rank\prms.dat

# path to your feature directory (ARFF files)
feat_dir=/dev/shm/

# path to Weka's jar file
# weka_jar=/opt/weka/weka.jar
weka_jar=/usr/share/java/weka.jar
test -f $weka_jar || exit -1

# memory to allocate for the JVM
jvm_mem=2048m

result_file=$path/_svm.res
rm $result_file


# java -Xmx1024m -classpath /usr/share/java/weka.jar  weka.filters.unsupervised.attribute.Normalize -i $trn_file -o $trn_file\2 
# java -Xmx1024m -classpath /usr/share/java/weka.jar  weka.filters.unsupervised.attribute.Normalize -i $tst_file -o $tst_file\2
# mv $trn_file\2 $trn_file
# mv $tst_file\2 $tst_file

# java -Xmx$jvm_mem -classpath /usr/share/java/weka.jar weka.classifiers.functions.SMO -o -K "weka.classifiers.functions.supportVector.PolyKernel -C 250007 -E 1.0" -t $trn_file -T $tst_file > $result_file

# java -Xmx$jvm_mem -classpath /usr/share/java/weka.jar weka.classifiers.meta.FilteredClassifier -o -c 19  -t $trn_file -T $tst_file -F "weka.filters.unsupervised.attribute.Remove -R 15,16,17,18" -W weka.classifiers.functions.SMO -- -K  "weka.classifiers.functions.supportVector.PolyKernel -C 250007 -E 1.0" > $result_file


# # # # # # # # # # # # # # 


# java -Xmx$jvm_mem -classpath /usr/share/java/weka.jar weka.classifiers.meta.FilteredClassifier -o -t $trn_file -T $tst_file -F "weka.filters.unsupervised.attribute.Remove -R 15,16" -W weka.classifiers.functions.SMO -- -K  "weka.classifiers.functions.supportVector.PolyKernel -C 250007 -E 1.0" > $result_file


# java -Xmx$jvm_mem -classpath /usr/share/java/weka.jar weka.classifiers.meta.FilteredClassifier -o -t $trn_file -T $tst_file -F "weka.filters.unsupervised.attribute.Remove -R 29,30" -W weka.classifiers.functions.SMO -- -K  "weka.classifiers.functions.supportVector.PolyKernel -C 250007 -E 1.0" > $result_file

# java -Xmx$jvm_mem -classpath /usr/share/java/weka.jar weka.classifiers.meta.FilteredClassifier -o -t $trn_file -T $tst_file -F "weka.filters.unsupervised.attribute.Remove" -W weka.classifiers.functions.SMO -- -K  "weka.classifiers.functions.supportVector.PolyKernel -C 250007 -E 1.0" > $result_file


java -Xmx$jvm_mem -classpath /usr/share/java/weka.jar:/usr/share/java/libsvm.jar weka.classifiers.meta.FilteredClassifier -v -o -t $trn_file -T $tst_file -F "weka.filters.unsupervised.attribute.Standardize" -W weka.classifiers.functions.LibSVM -- -S 0 -K 1 > $result_file


# # # # # # # # # # # # # 



# java -Xmx1024m -classpath /usr/share/java/weka.jar  weka.filters.unsupervised.attribute.Normalize -i trn_0__gafccf0.arff -o trn_0__gafccf0_n.arff 

# java -Xmx1024m -classpath /usr/share/java/weka.jar weka.classifiers.functions.SMO -o -t trn_0__gafccf0.arff -T tst_0__gafccf0.arff  

# -M -C 0.01 -L 0.0010 -P 1.0E-12 -N 0 -V -1 -W 1 -K "weka.classifiers.functions.supportVector.PolyKernel -C 250007 -E 1.0"

case $1 in
    show) cat $result_file ;;
    *) exit 1 ;;
esac


# java -Xmx2048m -classpath /usr/share/java/weka.jar weka.classifiers.lazy.IBk -t /home/leandro/.local/share/Trash/files/matlab2weka/iris.arff -split-percentage 80 -o -K 15


#  -S <int>
#   Set type of SVM (default: 0)
#     0 = C-SVC
#     1 = nu-SVC
#     2 = one-class SVM
#     3 = epsilon-SVR
#     4 = nu-SVR
#  
#  -K <int>
#   Set type of kernel function (default: 2)
#     0 = linear: u'*v
#     1 = polynomial: (gamma*u'*v + coef0)^degree
#     2 = radial basis function: exp(-gamma*|u-v|^2)
#     3 = sigmoid: tanh(gamma*u'*v + coef0)
#  
#  -D <int>
#   Set degree in kernel function (default: 3)
#  
#  -G <double>
#   Set gamma in kernel function (default: 1/k)
#  
#  -R <double>
#   Set coef0 in kernel function (default: 0)
#  
#  -C <double>
#   Set the parameter C of C-SVC, epsilon-SVR, and nu-SVR
#    (default: 1)
#  
#  -N <double>
#   Set the parameter nu of nu-SVC, one-class SVM, and nu-SVR
#    (default: 0.5)
#  
#  -Z
#   Turns on normalization of input data (default: off)
#  
