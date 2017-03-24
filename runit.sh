#!/bin/bash

for i in 1000 2000 4000 8000
do
    echo sync
    th main.lua $i
    echo "\nasync\n"
    th async.lua $i
done
