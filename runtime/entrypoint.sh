#!/bin/bash

exit_code=0

{
    cd /codeexecution

    echo "Unpacking submission..."
    unzip ./submission/submission.zip -d ./

    if [ -f "main.py" ]
    then
        echo "Running submission with Python"
        conda run -n py python main.py
    
    else
        echo "ERROR: Could not find main.py in submission.zip"
        exit_code=1
    fi

    echo "Exporting submission.csv result..."

    # Valid scripts must create a "submission.csv" file within the same directory as main
    if [ -f "submission.csv" ]
    then
        echo "Script completed its run."
        cp submission.csv ./submission/submission.csv
    else
        echo "ERROR: Script did not produce a submission.csv file in the main directory."
        exit_code=1
    fi

    echo "================ END ================"
} |& tee "/codeexecution/submission/log.txt"

# copy for additional log uses
cp /codeexecution/submission/log.txt /tmp/log
exit $exit_code
