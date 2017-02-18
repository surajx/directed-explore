export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:"`pwd`"/ALE"

# cd ./agent/Blob-PROST
# num_proc=`nproc`
# make clean
# make all -j$num_proc
# cd ../../

agent_conf="./agent/conf/bpro.cfg"

dt=`date +%H%M%S`
sysname=`hostname`
romLoc="./ALE/ROMS"
ckptLoc="./data/checkpoints"

for game_dir in $ckptLoc/*;
do
    for f in $game_dir/*.txt;
    do
        game=`echo $game_dir | awk 'BEGIN {FS="/"} {print $4}'`
        seed=$(( ( RANDOM % 10000 )  + 1486 ))
        c=`echo $f | awk 'BEGIN {FS="-Frames"} {print $1}'`.txt
        chkpt=`echo $f | awk 'BEGIN {FS="-checkPoint-Frames"} {print $1}'`
        cp $f $c
        echo "./agent/Blob-PROST/learnerBlobTime -s $seed -c $agent_conf -r $romLoc/$game.bin -n $chkpt"
        ./agent/Blob-PROST/learnerBlobTime -s $seed -c $agent_conf -r $romLoc/$game.bin -n $chkpt
        sleep 3
    done
done