root_path=/home/xintong/Documents/PyProjects/agileflight_policy/envtest/python
echo "root path: $root_path"

name_1="cr$1lv$2av$3"
path_1="$root_path/crcoeff_$1_lvcoeff_$2_avcoeff_$3/PPO_1"
echo "path 1: $path_1"

name_2="cr$4lv$5av$6"
path_2="$root_path/crcoeff_$4_lvcoeff_$5_avcoeff_$6/PPO_1"
echo "path 2: $path_2"

name_3="cr$7lv$8av$9"
path_3="$root_path/crcoeff_$7_lvcoeff_$8_avcoeff_$9/PPO_1"
echo "path 3: $path_3"

name_4="cr${10}lv${11}av${12}"
path_4="$root_path/crcoeff_${10}_lvcoeff_${11}_avcoeff_${12}/PPO_1"
echo "path 4: $path_4"

tensorboard --logdir_spec=$name_1:$path_1,$name_2:$path_2,$name_3:$path_3,$name_4:$path_4
