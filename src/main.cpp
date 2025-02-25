#include <iostream>
#include "Engine.hpp"

int main(){
    Engine e;
    e.init();    
    e.run();
    e.cleanup();
    return 0;
}