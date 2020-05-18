#!/usr/bin/perl



 $file =$ARGV[0];	# Nombra el fichero

 open(INFO, $file);		# Abre el fichero
 @lines = <INFO>;		# Lo lee en un array
 close(INFO);			# Cierra el fichero
 $c = @lines;
 $i = 0;
 
 while ($i<$c) {
    @linea = ' ';
    @linea = split(/\s+/,@lines[$i]);
    if ('UAR:' eq @linea[0]){
       chomp(@linea[1]);       
       
       $resu = @linea[1];
       $i = $c;
              
    }    
    $i++;
 }
 
 unlink($file);
 
 open(OUTF, ">$file");		# Abre el fichero 
 print OUTF $resu . " \n"; 
 close(OUTF);

 