#!/usr/bin/perl


 $ARGC = scalar @ARGV;

 if ($ARGC>0){
    $cromID = $ARGV[0];
 } else {
   $cromID = 0;
 }
#  $trn = "./data/GCM_Training.arff";
#  $tst = "./data/GCM_Test.arff";

 $trn = "./data/GCM_Training_TRN.arff";
 $tst = "./data/GCM_Training_TST.arff";

 if ($ARGC>0){
    system("python ./python/classificator.py -features $cromID -train $trn -test $tst > $cromID.res");
 } else {   
    system("python ./python/classificator.py -train $trn -test $tst > $cromID.res");
 }
 system("perl ./python/pruneresUAR.pl $cromID.res -train $trn");
 
 # -test $tst

 