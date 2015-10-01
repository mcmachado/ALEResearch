#include <ale_interface.hpp>
#include <iostream>
using namespace std;


int main(int argc, char** argv)
{
    ALEInterface ale(false);
    ale.loadROM(string(argv[1]));
    
    if(atoi(argv[2]) == 0){
        auto modes = ale.getAvailableModes();
        for(const auto& m : modes){
            cout<<(int)m<<endl;
        }
    }else{
        auto diff = ale.getAvailableDifficulties();
        for(const auto& d : diff){
            cout<<(int)d<<endl;
        }
    }

}
