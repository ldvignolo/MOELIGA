# include "dictionary.hpp"

void chop( std::string &s);

void chop( std::string &s)
{
  s.erase(s.find_last_not_of(" \n\r\t")+1);
};



//======================
// DICTIONARY CLASS
//======================

/* CONSTRUCTOR */

Dictionary::Dictionary(const std::string FILENAME)//const char *nombre_archivo)
{
  
  //==========================================
  // LEVANTO EL ARCHIVO
  //==========================================
  
  //  std::string FILENAME = "SETTINGS2.cfg";
  
  std::string linea="", key, value;
  std::size_t found;
  std::ifstream archivo(FILENAME.c_str());
  int next;
  
  if (archivo.is_open())
  {
      // while(archivo >> linea)
      while(getline(archivo, linea))
      {
        chop(linea);
	found = linea.find("=");
        if (found < (linea.size()-1))  
           next = found + 1;
        else
           next = found - 1;
        
	if ( (linea[0] != '#') && (found != std::string::npos) && (linea[next] != '=') )
	{
	  //----------------
	  // EXTRAIGO CLAVE
	  //----------------
	  key = linea.substr(0,found);
	  
	  
	  //----------------
	  // EXTRAIGO VALOR
	  //----------------
	  value = linea.substr(found+1,linea.size());
	  
	  
	  //------------------------
	  // DETERMINO TIPO DE DATO
	  //------------------------
	  
	  // BOLEANO --> Busco "True" or "False"
	  
	  /* TRUE */
	  if ( value.find("True") != std::string::npos )
	  {
	    KEYS[key] = value;
	    BOOLEAN[key] = true;
// // 	    std::cout <<"Type BOOLEAN --> " << key << " = " << BOOLEAN[key] << std::endl;
	  }
	  
	  /* FALSE */
	  else if ( value.find("False") != std::string::npos )
	  {
	    KEYS[key] = value;
	    BOOLEAN[key] = false;
// 	    std::cout <<"Type BOOLEAN --> " << key << " = " << BOOLEAN[key] << std::endl;
	  }
	  
	  
	  // STRING
	  else if ( value.find("\"") != std::string::npos )
	  {
	    KEYS[key] = value;
            found = value.find("\"");
            value = value.substr(found+1,value.size()-found);
            found = value.find("\"");
            value = value.substr(0,found);
            // cout << "<loadsettings> " << value << endl;
	    STRING[key] = value;  // evitamos comilla inicial y final
// 	    std::cout <<"Type STRING --> " << key << " = " << STRING[key] << std::endl;
	  }
	  
	  
	  // DOUBLE
	  else if ( value.find(".") != std::string::npos )
	  {
	    KEYS[key] = value;
	    DOUBLE[key] = std::stod(value);
// 	    std::cout <<"Type DOUBLE --> " << key << " = " << DOUBLE[key] << std::endl;
	  }
	  
	  // INTEGER
	  else 
	  {
	    KEYS[key] = value;
	    INTEGER[key] = std::stoi(value);
// 	    std::cout <<"Type INTEGER --> " << key << " = " << INTEGER[key] << std::endl;
	  }
	  
	}
	
      }
      archivo.close();
  }
  
  else
  {
    // show message:
    std::cout << "Error opening file.\n";
  }
  
  //==========================================
  
};



//==========================================
/* GET METHODS */
//==========================================


//==========================================
bool Dictionary::get_bool(std::string key)
{
  
  bool value = false;
  
  std::map<std::string, bool>::iterator it;
  
  it = this->BOOLEAN.find(key);
  
  if (it != this->BOOLEAN.end())
  {
    value = this->BOOLEAN[key];
  }
  
  return value;
  
};
//==========================================



//==========================================
int Dictionary::get_int(std::string key)
{
  
  int value = std::numeric_limits<int>::max() + 1;
  
  std::map<std::string, int>::iterator it;
  
  it = this->INTEGER.find(key);
  
  if (it != this->INTEGER.end())
  {
    value = this->INTEGER[key];
  }
  
  return value;
  
};
//==========================================



//==========================================
double Dictionary::get_dbl(std::string key)
{
  double value = std::numeric_limits<double>::infinity();
  
  std::map<std::string, double>::iterator it;
  
  it = this->DOUBLE.find(key);
  
  if (it != this->DOUBLE.end())
  {
    value = this->DOUBLE[key];
  }
  
  return value;
  
};
//==========================================



//==========================================
std::string Dictionary::get_str(std::string key)
{
  std::string value = "None";
  
  std::map<std::string, std::string>::iterator it;
  
  it = this->STRING.find(key);
  
  if (it != this->STRING.end())
  {
    value = this->STRING[key];
  }
  
  return value;
  
};
//========================================== 



//==========================================
std::map<std::string,std::string> Dictionary::get_key_value(void)
{
    return this->KEYS;
};
//==========================================
