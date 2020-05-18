#ifndef __DICTIONARY_H__
#define __DICTIONARY_H__

# include <stdio.h>
# include <stdlib.h>
# include <iostream>
# include <fstream>
# include <string>

# include <cstring>

# include <vector>
# include <map>

#include <limits>

//================================================================ 
class Dictionary
{
    private:
      
      std::map< std::string, std::string>      KEYS; // {key, map's name}
      std::map< std::string, bool>             BOOLEAN;
      std::map< std::string, int>              INTEGER;
      std::map< std::string, double>           DOUBLE;
      std::map< std::string, std::string>      STRING;
      

    public:
      
      //--------------
      // Constructor
      //--------------      
      Dictionary(const std::string FILENAME);
      
      std::map<std::string,std::string> get_key_value(void);
      
      bool get_bool(std::string key);
      int get_int(std::string key);
      double get_dbl(std::string key);
      std::string get_str(std::string key);

};

//==========================================================

#endif
 
