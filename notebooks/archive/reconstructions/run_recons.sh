#!/bin/bash

# example: ./run_recons.sh 0 4 coded_163c.conf
# ./run_recons.sh 0 17 coded_beads2.conf '--frame-mask 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 '

# /home/ubuntu/anaconda3/bin/python batch_reconstruction.py -c coded_beads2.conf --strip-index 0 

# aws s3 cp s3://motiondeblur/163a_color_coded_raster_motiondeblur_2018_05_22_21_47_35/ ~/datasets/163a_color_coded_raster_motiondeblur_2018_05_22_21_47_35 --recursive --region us-east-2

for ((i = $1; i <= $2; i++ )); do
    /home/ubuntu/anaconda3/bin/python batch_reconstruction.py -c $3 --strip-index $i $4
    aws s3 mv ~/reconstructions/ s3://motiondeblur/reconstructions/beads/ --recursive --region us-east-2 --grants full=id=f3e01911234678438f58d65607e76c55df61e5a3db5316a30394a24767e2ceb6 full=id=634b7a0686be3590c1808efc465ea9db660233386f1ad0bbbe3cabab19ae2564
done

# aws s3 sync s3://motiondeblur/reconstructions/beads/ ~/reconstructions --region us-east-1
# --recon-mode static  --save-tag static

