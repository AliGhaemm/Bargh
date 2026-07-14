#!/bin/bash

while true; do
    echo "Starting Python script..."
    pwd
    /home/hajali/Desktop/Bargh_Ml_project/BarghProject/bin/python /home/hajali/Desktop/Bargh_Ml_project/src/models/main.py &
    PYTHON_PID=$!
    echo "in running"

    while true; do
        if [[ -f condition.txt ]]; then
            CONDITION=$(cat condition.txt)
            if [[ "$CONDITINO" == "again" ]]; then
                echo "Condition met, runnig the code again."
                rm -f condition.txt
                sleep 10
                break
            elif [[ "$CONDITION" == "stop" ]]; then
                echo "Condition met, terminating the code."
                TERMINATING="true"
            fi
        fi
    done

    if [[ "$TERMANATING" == "true" ]]; then
        break
    fi

done