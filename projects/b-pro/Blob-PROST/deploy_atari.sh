#!/bin/bash

game_array=( venture montezuma_revenge freeway ) #frostbite qbert gravitar freeway bank_heist alien )
beta_array=( 0.00001 0.0001 0.00025 0.0005 0.00075 0.001 ) # 
dt=`date +%H%M`
loc=$1

for j in "${beta_array[@]}";
do
    # for j in `seq 1 3`;
    sed -i 's/^\(BETA\s*=\s*\).*$/\1'$j'/' ../conf/bpro.cfg
    for i in "${game_array[@]}";
    do
        seed=$(( ( RANDOM % 10000 )  + 1 ))
        ./learnerBlobTime -s $seed -c ../conf/bpro.cfg -r /home/users/u5881495/git/directed-explore/projects/Arcade-Learning-Environment/ROMS/$i.bin -w "$i"_phi_wts_"$loc"_beta$j -n "$i"_phi_wts_checkpoint_"$loc"_beta$j | gzip -9 > "$i"_phi_wts_"$loc"_beta"$j"_051016_"$dt".log.gzip &
    done
done
