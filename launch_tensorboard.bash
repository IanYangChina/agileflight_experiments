root_path=/home/xintong/Documents/PyProjects/agileflight_experiments/envtest/python
echo "root path: $root_path"

name_1="cr$1posx$2"
path_1="$root_path/collision_terminal_coeff_$1_posxcoeff_$2/PPO_1"
echo "path 1: $path_1"

name_2="cr$3posx$4"
path_2="$root_path/collision_terminal_coeff_$3_posxcoeff_$4/PPO_1"
echo "path 2: $path_2"

name_3="cr$5posx$6"
path_3="$root_path/collision_terminal_coeff_$5_posxcoeff_$6/PPO_1"
echo "path 3: $path_3"

name_4="cr${7}posx${8}"
path_4="$root_path/collision_terminal_coeff_${7}_posxcoeff_${8}/PPO_1"
echo "path 4: $path_4"

name_5="cr${9}posx${10}"
path_5="$root_path/collision_terminal_coeff_${9}_posxcoeff_${10}/PPO_1"
echo "path 5: $path_5"

name_6="cr${11}posx${12}"
path_6="$root_path/collision_terminal_coeff_${11}_posxcoeff_${12}/PPO_1"
echo "path 6: $path_6"

tensorboard --logdir_spec=$name_1:$path_1,$name_2:$path_2,$name_3:$path_3,$name_4:$path_4,$name_5:$path_5,$name_6:$path_6
