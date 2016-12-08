#!/bin/bash
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:"`pwd`"/ALE"
game_array=( montezuma_revenge freeway ) #venture frostbite qbert gravitar bank_heist alien )
beta_array=( 0.005 0.01 0.015 0.020 0.025 0.03 0.035 0.04 0.045 0.05 0.06 0.07 )
dt=`date +%H%M`
sysname=$1
romLoc="./ALE/ROMS"
end=$(($2 * 4 - 1))
start=$(($end - 3))

for j in "${beta_array[@]}:$start:$end";
do
    # for j in `seq 1 3`;
    sed -i 's/^\(BETA\s*=\s*\).*$/\1'$j'/' ./agent/conf/bpro.cfg
    for i in "${game_array[@]}";
    do
        seed=$(( ( RANDOM % 10000 )  + 1 ))
        #./learnerBlobTime -s $seed -c ./agent/conf/bpro.cfg -r $romLoc/$i.bin -w "$i"__"$sysname"_beta$j -n "$i"_checkpoint_"$sysname"_beta$j | xz -9 > "$i"_phi_wts_"$sysname"_beta"$j"_051016_"$dt".log.xz &
	./learnerBlobTime -s $seed -c ./agent/conf/bpro.cfg -r $romLoc/$i.bin -w "$i"__"$sysname"_beta$j -n "$i"_checkpoint_"$sysname"_beta$j > /dev/null &
    done
done