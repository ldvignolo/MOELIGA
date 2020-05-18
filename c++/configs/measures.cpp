#include "measures.hpp"
#include <algorithm>
#include <numeric>
#include <iostream>
#include <vector>
#include <functional>

//######################
//     STATISTICS
//######################

/*----------------------------
 * MAX 
 *----------------------------*/
double stt_max(std::vector<double> &v)
{
    std::vector<double>::iterator it = std::max_element(v.begin(),v.end());
    return *it;
};



/*----------------------------
 * MIN 
 *----------------------------*/
double stt_min(std::vector<double> &v)
{
    std::vector<double>::iterator it = std::min_element(v.begin(),v.end());
    return *it;
};

    

/*----------------------------
 * MEAN 
 *----------------------------*/
double stt_mean(std::vector<double> &v)
{

    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    double mean = sum / v.size();

    return mean;
};



/*----------------------------
 * DEVIATION ---> http://stackoverflow.com/questions/7616511/calculate-mean-and-standard-deviation-from-a-vector-of-samples-in-c-using-boos
 *----------------------------*/
double stt_std(std::vector<double> &v)
{
    double mean = 0.0;
    
    std::vector<double> diff(v.size(),0.0);
    
    std::transform(v.begin(), v.end(), diff.begin(), std::bind2nd(std::minus<double>(), mean));
    
    double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    
    double stdev = std::sqrt(sq_sum / v.size());
    
    return stdev;
};



/*----------------------------
 * MEDIAN 
 *----------------------------*/
double stt_median(std::vector<double> v)
{
    // this is middle for odd-length, and "upper-middle" for even length
    std::vector<double>::iterator middle = v.begin() + (v.end() - v.begin()) / 2;
    
    // This function runs in O(n) on average, according to the standard
    std::nth_element(v.begin(), middle, v.end());
    
    if ((v.end() - v.begin()) % 2 != 0) // odd length
    {
        return *middle;
    }
    else // even length
    { 
        // the "lower middle" is the max of the lower half
        std::vector<double>::iterator lower_middle = std::max_element(v.begin(), middle);
        return (*middle + *lower_middle) / 2.0;
    }
};



/*--------------------------------
 * MEDIAN ABSOLUTE DEVIATION (MAD)
 *-------------------------------*/
double stt_mad(std::vector<double> v)
{
    // CALCULO MEDIANA
    double median = stt_median(v);
    
    std::vector<double> diff(v.size(),0.0);
    
    std::transform(v.begin(), v.end(), diff.begin(), std::bind2nd(std::minus<double>(), median));
    
    //double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    //double mad = std::sqrt(sq_sum / v.size());
    
    /*=========================================
     * For a univariate data set X1, X2, ..., Xn,
     * the MAD is defined as the median of the
     * absolute deviations from the data's median:
     * 
     * MAD = median_i( |Xi - median_j(Xj)| )
     * 
     =========================================*/
    
    // APPLY ABSOLUTE VALUE
    for (auto& f : diff) { f = f < 0 ? -f : f;}
    
    double mad = stt_median(diff);
    
    return mad;
};

//############################################




//======================================
// CONSTRUCTOR
//======================================
Measures::Measures(int Gmax)
{
    // STARTING TIME
    this->start = std::chrono::system_clock::now();
    
    this->Gmax = Gmax;
    
    this->G = 0;
    
    this->idx_elite = 0;
    
    std::vector<double> vector(this->Gmax,0.0);
    this->elite["ABS_Nfeatures"] = vector;
    
    //-------------------------------------
    // DEFINO MEDIDA DE FITNES
    this->Set("Fitness");
    
    
    //-------------------------------------
    // DEFINO MEDIDA PARA CONTAR NUMERO DE FEATURES ACIVAS
    this->Set("Nfeatures");
    
    this->Set("ABS_Nfeatures"); 
    
    //-------------------------------------
    // DEFINO MEDIDA PARA LLEVAR REGISTRO DEL "UAR"
    this->Set("UAR");
    
        //-------------------------------------
    // DEFINO MEDIDA PARA LLEVAR REGISTRO DE LA DISTANCIA MEDIA ENTRE INDIDIVIDUOS
    this->Set("mDIST");
  
};



//======================================
// SET METHOD
//======================================
void Measures::Set(std::string measure)
{
    std::vector<double> vector(this->Gmax, 0);
    
    /* -------------------
       INDIVIDUO ELITE
       -------------------*/
    this->elite[measure] = vector;
    
    
    /* -------------------
       MEDIDAS GENERALES
       -------------------*/
    this->measures[measure]["max"] = vector;
    
    this->measures[measure]["min"] = vector;
    
    this->measures[measure]["mean"] = vector;
    
    this->measures[measure]["std"] = vector;
    
    this->measures[measure]["median"] = vector;
    
    this->measures[measure]["mad"] = vector;
};


//======================================
// GET METHOD [HISTORY] -- VECTOR OF DOUBLES
//======================================
std::vector<double> Measures::Get(std::string measure, std::string statistic)
{
    std::vector <double> v(1,0);
    
    if (this->measures.count(measure))
    {
        if (this->measures[measure].count(statistic))
        {
            std::vector<double> V(&this->measures[measure][statistic][0],&this->measures[measure][statistic][this->G]);
            return V;
        }
        else
        {
            std::cout << "Unknown statistic --> " << statistic << std::endl;
        }
    }
    
    else
    {
        std::cout << "Unknown measure --> " << measure << std::endl;
    }
    return v;
};



//======================================
// GET METHOD [HISTORY] -- DOUBLE
//======================================
double Measures::Get(std::string measure, std::string statistic, int G)
{
    if (this->measures.count(measure))
    {
        if (this->measures[measure].count(statistic))
        {
            return this->measures[measure][statistic][G];
        }
        else
        {
            std::cout << "Unknown statistic --> " << statistic << std::endl;
        }
    }
    
    else
    {
        std::cout << "Unknown measure --> " << measure << std::endl;
    }
    return 0;
};





//====================================================
// GET METHOD [ELITE INDIVIDUAL] -- VECTOR OF DOUBLES
//====================================================
std::vector<double> Measures::GetElite(std::string measure)
{
    std::vector <double> v(1,0);
    if (this->measures.count(measure))
    {
        std::vector<double> V(&this->elite[measure][0],&this->elite[measure][this->G]);
        return V;
    }
    else
    {
        std::cout << "Unknown measure --> " << measure << std::endl;
    }
    return v;
};



//==========================================
// GET METHOD [ELITE INDIVIDUAL] -- DOUBLE
//==========================================
double Measures::GetElite(std::string measure, int G)
{
    if (this->measures.count(measure))
    {
        return this->elite[measure][G];
    }
    else
    {
        std::cout << "Unknown measure --> " << measure << std::endl;
    }
    return 0;
};



//======================================
// UPDATE METHOD -- GLOBAL
//======================================
void Measures::Update(poblacion &P, int G)
{
    // ACTUALIZO CONTADOR DE GENERACIONES
    this->G = G;
    
    this->idx_elite = 0;
    
    double maxfitness = 0;
    
    
    //-------------------------------------
    // DEFINO MEDIDA DE FITNES
    std::vector<double> fitness(P.individuos.size(), 0.0);
    
    
    //-------------------------------------
    // DEFINO MEDIDA PARA CONTAR NUMERO DE FEATURES ACIVAS
    std::vector<double> Nfeatures(P.individuos.size(),0.0);      // valor de la funcion objetivo
    std::vector<double> ABSNfeatures(P.individuos.size(),0.0);   // cantidad real
    
    
    //-------------------------------------
    // DEFINO MEDIDA PARA LLEVAR REGISTRO DEL "UAR"
    std::vector<double> UAR(P.individuos.size(),0.0);
    
    //-------------------------------------
    // DEFINO MEDIDA PARA LLEVAR REGISTRO DE LA DISTANCIA MEDIA
    std::vector<double> meanDIST(P.individuos.size(),0.0);
    
    
    //---------------------------------------------
    // EXTRAIGO DE CADA INDIVIDUO DE LA POBLACION
    // LAS MEDIDAS A GUARDAR
    //---------------------------------------------
    double NF = 0;

    for(unsigned idx = 0; idx < P.individuos.size(); idx++)
    {
        fitness[idx] = P.individuos[idx].Fitness;
        
        UAR[idx] = P.individuos[idx].aptitud[0];
        
        Nfeatures[idx] = P.individuos[idx].aptitud[1];
        ABSNfeatures[idx] = std::count(P.individuos[idx].crom.begin(), P.individuos[idx].crom.end(), true);
        
        meanDIST[idx] = stt_mean(P.individuos[idx].distancias);
        
        if (maxfitness < fitness[idx])
        {
            this->idx_elite = idx; // ACTUALIZO EL INDICE DEL INDIVIDUO ELITE
            maxfitness = fitness[idx];
            NF = std::count(P.individuos[idx].crom.begin(), P.individuos[idx].crom.end(), true);
            
        }
        
    }
    
    this->elite["ABS_Nfeatures"][G] = NF;
    
    //-------------------------------------
    // ACTUALIZO EL NUMERO DE CARCATEIRSTICAS
    // DEL MEJOR INDIVIDUO
    // this->Update("Fitness", G, fitness);
    
    
    //-------------------------------------
    // ACTUALIZO ESTADISTICAS PARA FITNESS
    this->Update("Fitness", G, fitness);
    
    
    //-------------------------------------
    // ACTUALIZO ESTADISTICAS PARA NUMERO DE FEATURES
    this->Update("Nfeatures", G, Nfeatures);
    this->Update("ABS_Nfeatures", G, ABSNfeatures);
    
    //-------------------------------------
    // ACTUALIZO ESTADISTICAS PARA UAR
    this->Update("UAR", G, UAR);
    
    // ACTUALIZO ESTADISTICAS PARA mDIST
    this->Update("mDIST", G, meanDIST);
    
};




//======================================
// UPDATE METHOD -- VECTOR
//======================================
void Measures::Update(std::string measure, int G, std::vector<double> values)
{
    //===================================
    // INDIVIDUO ELITE
    //===================================
    this->elite[measure][G] = values[this->idx_elite];
    
    
    //===================================
    // MEDIDAS GENERALES
    //===================================
    
    std::vector<double>::iterator it;
    
    //--------------------
    // MAX
    //--------------------
    //it = std::max_element(values.begin(),values.end());
    this->measures[measure]["max"][G] = stt_max(values);//*it;
    
    //--------------------
    // MIN
    //--------------------
    //it = std::min_element(values.begin(),values.end());
    this->measures[measure]["min"][G] = stt_min(values); // *it;
    
    //--------------------
    // MEAN
    //--------------------
    //double sum = std::accumulate(values.begin(), values.end(), 0.0);
    //double mean = sum/values.size();
    this->measures[measure]["mean"][G] = stt_mean(values); // mean;
    
    
    //--------------------
    // DEVIATION
    // http://stackoverflow.com/questions/7616511/calculate-mean-and-standard-deviation-from-a-vector-of-samples-in-c-using-boos
    //--------------------
    //std::vector<double> diff(values.size());
    //std::transform(values.begin(), values.end(), diff.begin(), std::bind2nd(std::minus<double>(), mean));
    //double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    //double stdev = std::sqrt(sq_sum / values.size());
    this->measures[measure]["std"][G] = stt_std(values); // stdev;
    
    
    //--------------------
    // MEDIAN
    //--------------------
    //std::nth_element(values.begin(), values.begin() + values.size()/2, values.end());
    //double median = values[values.size()/2];
    this->measures[measure]["median"][G] = stt_median(values); // median;
    
    
    //--------------------
    // MAD
    //--------------------
    //std::transform(values.begin(), values.end(), diff.begin(), std::bind2nd(std::minus<double>(), median));
    //sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    //double mad = std::sqrt(sq_sum / values.size());
    this->measures[measure]["mad"][G] = stt_mad(values); // mad;
    
};






//======================================
// SAVE MEASURES
//======================================
void Measures::Save(const std::string FILENAME, Dictionary &SETTINGS)
{
    bool tab = true;  // INDENTACION DE LAS CLAVES EN EL STRING GENERADO
    
    //--------------
    /* NODO RAIZ */
    //--------------
    jsonlib json;
    
    
    //----------------------
    /* SAVE DATE AND TIME */
    //----------------------
    this->end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = this->end-this->start;
    std::time_t start_time = std::chrono::system_clock::to_time_t(this->start);
    std::time_t end_time = std::chrono::system_clock::to_time_t(this->end);
    
    json["TIME"]["start"] = std::ctime(&start_time);
    json["TIME"]["end"] = std::ctime(&end_time);
    json["TIME"]["elapsed time"] = elapsed_seconds.count();
    
    
    //-----------------
    /* SAVE SETTINGS */
    //-----------------
    json["SETTINGS"] = SETTINGS.get_key_value();
    
    
    
    //-------------------------
    /* SAVE MEASURES */
    //-------------------------
    // Update measures to be saved until "this->G"
    
    json["MEASURES"]["Generations"] = this->G;
    
    
    //--------------------------------
    /* SAVE MEASURES DELETING ZEROS */
    //--------------------------------
    
    
    /*  */
    std::vector<double> V;
    
    std::map<std::string, std::vector<double> >::iterator it1;
    std::map<std::string, std::map<std::string, std::vector<double> > >::iterator it2;
    
    //---------------------------
    /* SAVE MEASURES [ELITE] */
    //---------------------------
    for (it1 = this->elite.begin(); it1 != this->elite.end(); it1++)
    {
        V = this->elite[it1->first];
        V.resize(this->G+1);
        json["MEASURES"]["ELITE"][it1->first] = V;
    }
    
    
    
    //---------------------------
    /* SAVE MEASURES [GENERAL] */
    //---------------------------
    for (it2 = this->measures.begin(); it2 != this->measures.end(); it2++)
    {        
        
        for (it1 = this->measures[it2->first].begin(); it1 != this->measures[it2->first].end(); it1++)
        {
            V = this->measures[it2->first][it1->first];
            V.resize(this->G+1);
            json["MEASURES"]["GENERAL"][it2->first][it1->first] = V;
        }
    }
    
    
    
    
    
    
    //--------------------
    /* EXPORT TO STRING */
    //--------------------
    std::string json_string;
    if (tab == true)
    {
        json_string = json.dump(4);
    }
    else
    {
        json_string = json.dump();
    }
    //std::cout << json_string << std::endl;
    
    
    //-------------
    /* SAVE DATA */
    //-------------
    std::ofstream file(FILENAME.c_str());
    file << json_string;
    file.close();
  
}; 
