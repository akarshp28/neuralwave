#!/bin/bash

nodes=$(pbsnodes | grep "properties = $1" -B 5 | grep "uncc")
echo $nodes

node=$(echo $nodes | cut -d' ' -f$2)

get_gpu_field() {
    dat=$(echo $data | cut -d',' -f$1 | cut -d';' -f$2 | cut -d'=' -f2)
    echo -n $dat
}

header="\n %-17s %20s %15s\n"
width=55
divider===============================
divider=$divider$divider

print_data() {
    data=$(pbsnodes $node | grep "gpu_status")
    num_gpus=$(echo $data | grep -o "gpu\[" | wc -l)

    printf "$header" "GPU Name" "Memory-Usage" "GPU-Util"
    printf "%$width.${width}s\n" "$divider"

    for (( value=1; value<=$num_gpus; value++ ))
    do
        printf "%d %18s %10s/%8s      %5s\n" $value "$(get_gpu_field $value 4)" "$(get_gpu_field $value 7)" "$(get_gpu_field $value 6)" "$(get_gpu_field $value 10)"
    done
}

time=1

if [ $3 = "-l" ]
then
    time=$4
fi

clear
while true
do
    print_data
    sleep $time
    clear
done
