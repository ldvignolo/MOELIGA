#!/usr/bin/perl

# A tool for making data partition from ARFF files.
# Author: Leandro Vignolo
#         ldvignolo@sinc.unl.edu.ar
#         26/10/2015




 use File::Basename;
 use List::Util qw(shuffle);
 use POSIX;


 $ARGC = scalar @ARGV;
 
 if ($ARGC==0){
   
   print("USAGE:  perl kfold.pl <infile.arff> -i <index_file.idx> -trnfile train.arff -tstfile test.arff -k nfolds -fold nfold \n ");
   print("> -fold 1 : initializes index file. \n ");
   exit();
 }

 $INfile = $ARGV[0];

 ($base, $dir, $ext) = fileparse("$INfile", qr/\.[^.]*/); 
 
 for ($i=1;$i<$ARGC;$i++){ 
     if ($ARGV[$i] eq '-fold' ){
         $fold = $ARGV[$i+1];
     }     
 } 
 
 $nfolds = 10;
 $fold = 1;
 $trnfile = "$base.train.$fold$ext";
 $tstfile = "$base.test.$fold$ext";
 $idxfile = "kfold_index.idx";
 $upsample = 0;
 $verb = 0;

 
 for ($i=1;$i<$ARGC;$i++){
 
     if ($ARGV[$i] eq '-k' ){
         $nfolds = $ARGV[$i+1];
     }
     
     if ($ARGV[$i] eq '-i' ){
         $idxfile = $ARGV[$i+1];
     }
     
     if ($ARGV[$i] eq '-fold' ){
         $fold = $ARGV[$i+1];
         if ($fold == 1) {
            unlink("$idxfile");
         }
     }
     if ($ARGV[$i] eq '-trnfile' ){
         $trnfile = $ARGV[$i+1];
     }
     if ($ARGV[$i] eq '-tstfile' ){
         $tstfile = $ARGV[$i+1];
     }     
   
 }
 

 open(INFO, $INfile);
 @lines = <INFO>;
 close(INFO);
 $c = @lines;
 $i = 0;
 
 $NH = 0;
 $na = 0;
 
 open(TRN, ">$trnfile");
 open(TST, ">$tstfile");

 while ($i<$c) {
    @linea = ' ';
    @linea = split(/\s+/,@lines[$i]);
    
    if (('@ATTRIBUTE' eq @linea[0])||('@Attribute' eq @linea[0])||('@attribute' eq @linea[0])){
       $na++;
    }
    
    if (('class' eq @linea[1])||('CLASS' eq @linea[1])||('Class' eq @linea[1])){
    
       @linea = split(/{/,@lines[$i]);
       @linea = split(/}/,$linea[1]);
       @clases = split(/,/,$linea[0]);
       
       $nc = scalar @clases;
       $ni = $na-1;
       for ($j=0;$j<$nc;$j++){
           $clases[$j] =~ s/^\s+|\s+$//g;
           # print(" > .$clases[$j]. \n");
       } 
    } 
    
    if (('@DATA' ne @linea[0])&&('@data' ne @linea[0])){
       
       print TRN $lines[$i];
       print TST $lines[$i];
       
    } else {
      print TRN $lines[$i];
      print TST $lines[$i];
      
      $NH = $i+1;
      $i = $c;
    }
    $i++;
 }
 close(TRN); 
 close(TST);
 
 
 my $np = 0;
  
 for ($i=$NH;$i<$c;$i++){
    @linea = ' ';
    @linea = split(/,/,@lines[$i]);
    # print("$linea[0] $linea[1] $linea[2] $linea[3]  \n");
    
    chomp($linea[$ni]);
    $q = scalar @linea;
    
    if ( $q > 0){
      $np=$np+1;          
    }
           
 }
 
 $nbatch = ceil($np/$nfolds); 
 
 $nTST = $nbatch;
 $nTRN = $np-$nbatch; 
 
 $Ntot = $np;
 
 my @indices = (0..($Ntot-1));
 my @IDX;
 
 if ($fold==1)
 {
    @IDX = shuffle @indices;
    open(INFO, ">$idxfile");
    for ($j=0;$j<$Ntot;$j++){
      print INFO "$IDX[$j]\n";
      }
    close(INFO);
   
 } 
 
 
 
 if (($fold!=1)&&(-e "$idxfile")) 
 {
   open(INFO, $idxfile);
   @itmp = <INFO>;
   close(INFO);
   $inp = @itmp;   
   if ($Ntot!=$inp){
       print("Error:  index file not compatible. \n ");
       exit();
   }
   
   for ($j=0;$j<$Ntot;$j++){
       chomp($itmp[$j]);
       $IDX[$j] = $itmp[$j];
   }
 }
 if (($fold!=1)&&(!(-e "$idxfile")))
 {
      print("Error:  index file not found. \n ");
      exit();  
 }
 
 open(TRN, ">>$trnfile"); 
 for ($j=0;$j<($fold-1)*$nbatch;$j++){
      
        $ii = $NH+$IDX[$j];
	chomp($lines[$ii]);
	print TRN "$lines[$ii]\n";
 }
 
 if ($fold < $nfolds)
 {
    for ($j=$fold*$nbatch;$j<$Ntot;$j++){
	  
	    $ii = $NH+$IDX[$j];
	    chomp($lines[$ii]);
	    print TRN "$lines[$ii]\n";
    }
 }
 close(TRN);
 

 open(TST, ">>$tstfile"); 

 if ($fold == $nfolds)
 {
   $End = $Ntot; 
 } else {
   $End = $fold*$nbatch;
 }

 for ($j=($fold-1)*$nbatch;$j<$End;$j++){
   
     $ii = $NH+$IDX[$j];
     chomp($lines[$ii]);
     print TST "$lines[$ii]\n";
 # print(">> TST $ii \n");
 }
 close(TST);
 
 
 