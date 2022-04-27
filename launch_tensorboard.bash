root_path=/home/xintong/Documents/PyProjects/agileflight_experiments/envtest/python
echo "root path: $root_path"

level="medium_"
name_1="crt$1cr$2posx$3"
path_1="$root_path/${level}collision_terminal_coeff_$1_crcoeff_$2_posxcoeff_$3/PPO_1"
echo "path 1: $path_1"

name_2="crt$4cr$5posx$6"
path_2="$root_path/${level}collision_terminal_coeff_$4_crcoeff_$5_posxcoeff_$6/PPO_1"
echo "path 2: $path_2"

name_3="crt$7cr$8posx$9"
path_3="$root_path/${level}collision_terminal_coeff_$7_crcoeff_$8_posxcoeff_$9/PPO_1"
echo "path 3: $path_3"

name_4="crt${10}cr${11}posx${12}"
path_4="$root_path/${level}collision_terminal_coeff_${10}_crcoeff_${11}_posxcoeff_${12}/PPO_1"
echo "path 4: $path_4"

name_5="crt${13}cr${14}posx${15}"
path_5="$root_path/${level}collision_terminal_coeff_${13}_crcoeff_${14}_posxcoeff_${15}/PPO_1"
echo "path 5: $path_5"

name_6="crt${16}cr${17}posx${18}"
path_6="$root_path/${level}collision_terminal_coeff_${16}_crcoeff_${17}_posxcoeff_${18}/PPO_1"
echo "path 6: $path_6"

tensorboard --logdir_spec=$name_1:$path_1,$name_2:$path_2,$name_3:$path_3,$name_4:$path_4,$name_5:$path_5,$name_6:$path_6
