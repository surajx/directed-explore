#!/bin/bash
if [ "$#" -ne 3 ]; then
    echo "Illegal number of parameters"
    echo "Usage:"
    echo "./deploy_phi_explore.sh <system_name> <beta_block> <trials_per_game>"
    exit
fi

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:"`pwd`"/ALE"

game_array=( montezuma_revenge freeway ) #venture frostbite qbert gravitar bank_heist alien )
beta_array=( 0.005 0.01 0.015 0.02 0.025 0.03 0.035 0.04 0.045 0.05 0.06 0.07 )

agent_conf="./agent/conf/bpro.cfg"

dt=`date +%H%M%S`
sysname=$1
num_cores=`lscpu | grep -i Core | grep -i "per socket" | awk '{ print $4}'`
romLoc="./ALE/ROMS"

eff_coref=$(($num_cores / ${#game_array[@]}))
end=$(($2 * $eff_coref))
start=$(($end - $eff_coref))
beta_block=("${beta_array[@]:$start:$end}")

repeat_game=$3

echo "system name: "$sysname
echo "Num Cores: "$num_cores
echo "Beta Block: "${beta_block[@]}
echo "Game Block: "${game_array[@]}
echo "No. experiments per game: "$repeat_game
echo "Agent configuration file: "$agent_conf
cat $agent_conf
echo "**Please verify and note these details down**"
read -p "Are you sure you want to deploy[Yy/Nn]? " -n 1 -r
echo    # (optional) move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]
then
    for beta in "${beta_block[@]}";
    do
        sed -i 's/^\(BETA\s*=\s*\).*$/\1'$beta'/' $agent_conf
        for game in "${game_array[@]}";
        do
            for trial in `seq 1 $repeat_game`;
            do
                seed=$(( ( RANDOM % 10000 )  + 1486 ))                
                echo "[DEPLOY::$beta] ./agent/Blob-PROST/learnerBlobTime -s $seed -c $agent_conf -r $romLoc/$game.bin -w "$game"__"$sysname"_beta_"$beta"_trail_"$trial" -n "$game"__"$sysname"_beta_"$beta"_trial_"$trial"_chkpt > /dev/null &"
                ./agent/Blob-PROST/learnerBlobTime -s $seed -c $agent_conf -r $romLoc/$game.bin -w "$game"__"$sysname"_beta_"$beta"_trail_"$trial" -n "$game"__"$sysname"_beta_"$beta"_trial_"$trial"_chkpt > /dev/null &
            done
        done
    done
    
    #rollback conf file change to beta=0.05
    sed -i 's/^\(BETA\s*=\s*\).*$/\1'0.05'/' $agent_conf
fi