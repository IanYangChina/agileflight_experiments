root_path=/home/xintong/Documents/PyProjects/agileflight_experiments/envtest/python
echo "root path: $root_path"

lv=0.0
av=0.0

name_1="cr$1lv${lv}posx$2av${av}"
path_1="$root_path/crcoeff_$1_lvcoeff_${lv}_posxcoeff_$2_avcoeff_${av}/PPO_1"
echo "path 1: $path_1"

name_2="cr$3lv${lv}posx$4av${av}"
path_2="$root_path/crcoeff_$3_lvcoeff_${lv}_posxcoeff_$4_avcoeff_${av}/PPO_1"
echo "path 2: $path_2"

name_3="cr$5lv${lv}posx$6av${av}"
path_3="$root_path/crcoeff_$5_lvcoeff_${lv}_posxcoeff_$6_avcoeff_${av}/PPO_1"
echo "path 3: $path_3"

name_4="cr${7}lv${lv}posx${8}av${av}"
path_4="$root_path/crcoeff_${7}_lvcoeff_${lv}_posxcoeff_${8}_avcoeff_${av}/PPO_1"
echo "path 4: $path_4"

name_5="cr${9}lv${lv}posx${10}av${av}"
path_5="$root_path/crcoeff_${9}_lvcoeff_${lv}_posxcoeff_${10}_avcoeff_${av}/PPO_1"
echo "path 5: $path_5"

name_6="cr${11}lv${lv}posx${12}av${av}"
path_6="$root_path/crcoeff_${11}_lvcoeff_${lv}_posxcoeff_${12}_avcoeff_${av}/PPO_1"
echo "path 6: $path_6"

tensorboard --logdir_spec=$name_1:$path_1,$name_2:$path_2,$name_3:$path_3,$name_4:$path_4,$name_5:$path_5,$name_6:$path_6
