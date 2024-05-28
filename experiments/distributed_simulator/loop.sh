#!/bin/bash
for i in {1..1}
do
    for j in {1..2}
    do
        python3.9 node${j}/main.py &
    done

    flag=0
    # while [ $flag -eq 0 ]; do
    pid=$(pgrep python3.9)
    while [ -n "$pid" ]
    do
        echo "Found a matching process with PID $pid"
    done
  
        # echo "$pid" | while read line; do
          
            # if [ "$line" = "10394" ]; then
            #     flag=1
            # fi
            
            # if [ "$line" != "10394" ]; then
                
            #     flag=0
            # fi
        

    # done
    echo "loop finish"

    # if [ -n "$pid" ]; then
    #     echo "Found a matching process with PID $pid"
    # else
    #     echo "No matching process found"
    # fi
    # if [ ${?} -eq 0 ]; then
    # echo "Found a matching process"
    # else
    # echo "No matching process found"
    # fi

done



