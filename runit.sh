#!/bin/bash

for i in 1000 2000 4000 8000
do
    echo
    echo sync
    th main.lua $i
    echo
    echo async
    th async.lua $i
done
