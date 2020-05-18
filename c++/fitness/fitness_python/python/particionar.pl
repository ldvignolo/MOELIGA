#!/usr/bin/perl

# A tool for making data partition from ARFF files, allowing to upsample clases in order to provide class balance.
# Author: Leandro Vignolo
#         ldvignolo@sinc.unl.edu.ar
#         01/07/2015




 use File::Basename;
 use List::Util qw(shuffle);
 use POSIX;


 $ARGC = scalar @ARGV;
 
 if ($ARGC==0){
   
   print("USAGE:  perl particionar.pl <infile.arff> -ptrain 80 -trnfile train.arff -tstfile test.arff -upsample 1 -verbose \n ");
   exit();
 }

 $INfile = $ARGV[0];

 ($base, $dir, $ext) = fileparse("$INfile", qr/\.[^.]*/); 
 
 # print("> $INfile = $base, $dir, $ext  \n");
 
 $ptrain = 80;
 $trnfile = "$base.train$ext";
 $tstfile = "$base.test$ext";
 $upsample = 0;
 $verb = 0;
 
 for ($i=1;$i<$ARGC;$i++){
 
     if ($ARGV[$i] eq '-ptrain' ){
         $ptrain = $ARGV[$i+1];
     }
     if ($ARGV[$i] eq '-trnfile' ){
         $trnfile = $ARGV[$i+1];
     }
     if ($ARGV[$i] eq '-tstfile' ){
         $tstfile = $ARGV[$i+1];
     }
     if ($ARGV[$i] eq '-upsample' ){
         $upsample = $ARGV[$i+1];
     }
     if ($ARGV[$i] eq '-verbose' ){
         $verb = 1;
     } 
     
 }
 
 
 if ($ptrain<=1){
    $ptrain = $ptrain*100;
 }
 
 if ($ptrain<=100){
    $ptest = 100 - $ptrain;
 } else {
    $ptest = 0;
    $ptrain = 100;
 }
 
 if ($verb){
    print("> Input file: $INfile \n");
    print("> Output training: $trnfile \n");
    print("> Output test: $tstfile \n");
 }
 
 

 open(INFO, $INfile);
 @lines = <INFO>;
 close(INFO);
 $c = @lines;
 $i = 0;
 
 $NH = 0;
 $na = 0;
 
 open(TRN, ">$trnfile");
 if ($ptest>0){
    open(TST, ">$tstfile");
 }
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
       if ($ptest>0){
	  print TST $lines[$i];
       }
       
    } else {
      print TRN $lines[$i];
      if ($ptest>0){
	  print TST $lines[$i];
      }
      $NH = $i+1;
      $i = $c;
    }
    $i++;
 }
 close(TRN);
 if ($ptest>0){
    close(TST);
 }
 
 my @nxc = (0) x $nc; 
  
 for ($i=$NH;$i<$c;$i++){
    @linea = ' ';
    @linea = split(/,/,@lines[$i]);
    # print("$linea[0] $linea[1] $linea[2] $linea[3]  \n");
    
    chomp($linea[$ni]);
    $linea[$ni] =~ s/^\s+|\s+$//g;
    
    for ($j=0;$j<$nc;$j++){
       if ($linea[$ni] eq $clases[$j]){
          # $IDX saves pattern's line number per each class
          $IDX[$j][$nxc[$j]] = $i;
          # $nxc saves number of patterns for each class
          $nxc[$j]=$nxc[$j]+1;          
       }
    }       
 }
 
 if ($verb){
    print(  "> Original class counts: ");
    for ($j=0;$j<$nc;$j++){
	  
	print("$clases[$j] $nxc[$j]; ");
	  
    } print("\n");
 }

 for ($j=0;$j<$nc;$j++){
      
     $nxcTST[$j] = ceil($nxc[$j]*($ptest/100));
     if ($nxcTST[$j]==0) # means near one instance available
     {
         $nxcTST[$j] = ceil(rand());   # use it for test or test randomly
     }
     $nxc[$j] = $nxc[$j] - $nxcTST[$j]; 
 }
 
 
 for ($j=0;$j<$nc;$j++){    
      for ($k=0;$k<$nxcTST[$j];$k++){
      
	  $IDxTST[$j][$k] = $IDX[$j][$k+$nxc[$j]]; 
      }
 }
 
 
 my $idMax = 0;
    
 $nxc[$idMax] > $nxc[$_] or $idMax = $_ for 1 .. $#nxc;
 
 $MaxCount = $nxc[$idMax];

 for ($j=0;$j<$nc;$j++){
 
    if ($j!=$idMax){
    
        for ($k=$nxc[$j];$k<$MaxCount;$k++){
            $IDX[$j][$k] = $IDX[$j][mod($k,$nxc[$j])];            
        }
    } 
 }
 
 sub mod { return int( $_[0]/$_[1]) , ($_[0] % $_[1]); }

 $Ntot = $MaxCount * $nc;
 
 my @indices = (0..($Ntot-1));
 my @sufle = shuffle @indices;
 
 for ($j=0;$j<$nc;$j++){
     $nxcTRN[$j] = 0;
 }    
 
 open(TRN, ">>$trnfile"); 
 for ($j=0;$j<$Ntot;$j++){
      
     $l = int($sufle[$j]/$nc);
     $k = mod($sufle[$j],$nc);

     if (($upsample==1)||($l<$nxc[$k])){
	$ii = $IDX[$k][$l];
	chomp($lines[$ii]);
	print TRN "$lines[$ii]\n";
	$nxcTRN[$k] = $nxcTRN[$k] + 1;
     }
 
 }
 close(TRN);
 
 if ($ptest>0){
    open(TST, ">>$tstfile"); 
    for ($j=0;$j<$nc;$j++){
	for ($l=0;$l<$nxcTST[$j];$l++){ 
    
	    $ii = $IDxTST[$j][$l];
	    chomp($lines[$ii]);
	    print TST "$lines[$ii]\n";
	}    
    }
    close(TST);
 }
 
 if ($verb){
    print(  "> Training class counts: ");
    for ($j=0;$j<$nc;$j++){      
	print("$clases[$j] $nxcTRN[$j]; ");      
    } print("\n");
    print(  "> Test class counts:     ");
    for ($j=0;$j<$nc;$j++){      
	print("$clases[$j] $nxcTST[$j]; ");      
    } print("\n");
    print("\n");
 }
 