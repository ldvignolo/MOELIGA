//----------------------------------------------------------------------------
//
//  Función de Fitness para la versión paralela
//
//  Este programa es llamado desde ag_lin_p, como
//  ejecutable a parte para correr en cada nodo (MPI_COMM_spawn).
//
//
//  Esta version es para utilizar con los datos ASM de caras (proyecto brasil)
//  Miércoles 21 Marzo 2012
//
// 
//  26-02-13 - Adaptacion para el challenge Interspeech ComParE
// 
//  Leandr0
//
//----------------------------------------------------------------------------


#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <float.h>
#include <vector>

#undef SEEK_SET
#undef SEEK_END
#undef SEEK_CUR

#include <mpi.h>

#include "types.h"

using namespace std;

double fitness(cromosoma crom, int lbits, int rank, float seed, short pobtype);

int main(int argc, char** argv)
{
        int nlcrom,lcrom = 10;   // valor para inicializar, recibo valor verdadero por MPI
	                      
	int id = atoi(argv[1]);
	cromosoma cromovect;
	double aptitud;
	int i;
	int params[2];
	float seed;
	short pobtype;

	int rank, size;
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);

        MPI_Status status;
	MPI_Comm parent_comm;
	MPI_Comm_get_parent(&parent_comm);

	int* cromo = (int*) malloc(lcrom * sizeof(int));
	
	cromo[0]=10;

	while (cromo[0]!=-5)
	{

		MPI_Recv(params, 2, MPI_INTEGER, 0, id, parent_comm, &status);
		MPI_Recv(&seed, 1, MPI_FLOAT, 0, id, parent_comm, &status);

		nlcrom = params[0];
		pobtype = params[1];

		if (nlcrom!=lcrom){
		    lcrom = nlcrom;
		    cromo = (int*) realloc(cromo, lcrom * sizeof(int)); 
		}

		MPI_Recv(cromo, lcrom, MPI_INTEGER, 0, id, parent_comm, &status);

		cromovect.resize(0);
		for (i=0;i<lcrom;i++){
			if (cromo[i]==1) 
			  cromovect.push_back(true);
			else 
			  cromovect.push_back(false);
		}

		if (cromo[0]!=-5){
			aptitud = fitness(cromovect, lcrom, rank, seed, pobtype); 
			MPI_Send(&aptitud, 1, MPI_DOUBLE, 0, id, parent_comm);
		}

	}
	                                                                                                            
	free(cromo);
	MPI_Finalize();

	return 0;

}



double fitness(cromosoma crom, int lbits, int rank, float seed, short pobtype)
{
     string cadena, cadena1, aux;
     double reconoc;
     int Lcrom = crom.size();

     if (Lcrom!=lbits)
     {
        cout << ">> Error en el tamaño del cromosoma <<" << endl;
        return 0;
     }

     /************************************************************************************/

     // crear el archivo de texto con una lista de los parametros
     string rnk, fname = "prms.dat";
     rnk=itoa(rank,10);
     if (rnk == "") rnk.insert(0, "0");
     fname.insert(0, rnk.c_str());
     ofstream fparams(fname.c_str(), ios::out);
     int cparams = 0;
     int y=0;
     for (int k=0;k<Lcrom;k++)
     {  
         if (crom[k]){
	   
	   if (cparams>0) fparams << ",";
	   cparams++;
	   y=k+1;
	   fparams << y;
	   
	 }  
     }

     fparams.close();
    
     // llamar al script pasando el id del nodo

     string cmd = rnk;

     cmd = rnk;
     
     if (pobtype == 1){
        cmd.insert(0,"sh weka/wekasvmcv.sh ");
     } else if (pobtype == 2){
        cmd.insert(0,"sh weka/wekasvmcv.sh ");
     }
     system(cmd.c_str());
     
     // leer resultado de archivo temporal
     cmd = ".res";
     cmd.insert(0,rnk);
     ifstream res(cmd.c_str(), ios::in);
     res >> reconoc;
     res.close();

     double penalizacion = (double (Lcrom-cparams))/Lcrom;
     
     reconoc = (double) 0.9*(reconoc/100) + 0.1*penalizacion;

     return reconoc;
     
}


