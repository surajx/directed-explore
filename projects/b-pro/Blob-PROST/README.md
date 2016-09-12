# Howto get it running

* Fork ALE from https://github.com/mcmachado/Arcade-Learning-Environment
* Fork B_PRO from https://github.com/mcmachado/b-pro

#### Configuring ALE

* makefile.unix
    * SET: USE_SDL=1
    * CMD: make all -f makefile.unix -j8

#### Configuring B-PRO

* b-pro/Blob-PROST/mainBlobTime.cpp
    * CHANGE
        * -ale.setFloat("frame_skip", param.getNumStepsPerAction());
        * +ale.setInt("frame_skip", param.getNumStepsPerAction());

* b-pro/conf/bpro.cfg
    * SET: DISPLAY = 1

* Copy ale.cfg from ALE repo to b-pro/Blob-PROST

* SYSTEM
    * SET: LD_LIBRARY_PATH="$LD_LIBRARY_PATH:<path to ale>"

* Makefile
    * SET: USE_SDL := 1
    * CMD: make all

#### Run the game

* ./learnerBlobTime -s 3434 -c ../conf/bpro.cfg -r /home/surajx/git/gym/env/lib/python2.7/site-packages/atari_py/atari_roms/montezuma_revenge.bin
