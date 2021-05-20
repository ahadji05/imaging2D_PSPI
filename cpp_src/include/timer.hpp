#ifndef TIMER_HPP
#define TIMER_HPP

#include <iostream>
#include <string>
#include <chrono>

extern "C"
{

//timer uses milliseconds to catch floating point precision later
//when we want to summarize all timings before exit in seconds.
class timer {

    public:
        std::chrono::system_clock::time_point clickStart;
        std::chrono::system_clock::time_point clickStop;
        std::chrono::duration<double, std::milli> elapsed;
        std::string name;
        int ncalls;
        double t_sum_milli;
        double t_sum_seconds;

        timer(const std::string & name) : name(name){
            ncalls = 0;
            t_sum_milli = 0.0;
            t_sum_seconds = 0.0;
        }

        ~timer(){};

        void start(){
            ncalls += 1;
            clickStart = std::chrono::high_resolution_clock::now();
        }

        void stop(){
            clickStop = std::chrono::high_resolution_clock::now();
            elapsed = clickStop - clickStart;
            t_sum_milli += (double)(elapsed.count());
            t_sum_seconds = t_sum_milli / 1000.0;
        }

        void dispInfo(){
            std::cout << name << ": ";
            std::cout << "calls(" << ncalls << ") total ";
            std::cout << t_sum_seconds << " seconds." << std::endl;
        }

};

} // end extern C

#endif