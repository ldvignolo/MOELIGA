#!/usr/bin/perl

# Script para extraer el mejor cromosoma (el ultimo que aparece en el txt)
# Abre todos los txt en el dir actual, o en el dir que se le pasa como argumento, y los guarda en el subdir ./croms



use strict;
use warnings;



my $ARGC=scalar @ARGV;
my $dir="./";

if ($ARGC>=1) {
   $dir=$ARGV[0];
}


opendir(DIR, $dir) or die $!;

my $i=0;
my @flist;

while (my $file = readdir(DIR)) {

    # We only want files
    next unless (-f "$dir/$file");

    # Use a regular expression to find files ending in .txt
    next unless ($file =~ m/\.txt$/);
    next unless ($file =~ m/AG/);

    $flist[$i]=$file;
    $i=$i+1;
}
my $N=$i;

closedir(DIR);

my $PATH="./croms";

mkdir("$dir/$PATH");

for ($i=0;$i<$N;$i++){

  print "-> $flist[$i]\n";
  &explorefile($flist[$i],$dir,$PATH);

}





sub explorefile {

 my $file = $_[0];
 my $dir  = $_[1];
 my $path = $_[2];

 my @kk = split(/\.txt/,$file);
 my $out = "$dir/$path/$kk[0]\.crom";


 open(INFO, "$dir/$file");	# Abre el fichero
 open(OUTF, ">$out");		# Abre el fichero
 my @lines = <INFO>;		# Lo lee en un array
 close(INFO);			# Cierra el fichero
 my $c = @lines;
 my $i=0;
 my @linea;
 my $TMP;
 my $len;
 
 while ($i<$c) {
    @linea = ' ';
    @linea = split(/\s+/,$lines[$i]);
    $len = @linea;
    if ($len>=1){
      if (('::>' eq $linea[0])&&('Coeficientes' eq $linea[1])&&('Seleccionados:' eq $linea[2])){
      
	$TMP = $lines[$i+1];
	
      }
    }
    $i++;
 }
 print OUTF $TMP . " \n";
 close(OUTF);

}
