root_path=/home/xintong/Documents/PyProjects/agileflight_experiments/envtest/python
echo "root path: $root_path"

lv=-0.01

name_1="cr$1lv${lv}posx$2av$3"
path_1="$root_path/crcoeff_$1_lvcoeff_${lv}_posxcoeff_$2_avcoeff_$3/PPO_1"
echo "path 1: $path_1"

name_2="cr$4lv${lv}posx$5av$6"
path_2="$root_path/crcoeff_$4_lvcoeff_${lv}_posxcoeff_$5_avcoeff_$6/PPO_1"
echo "path 2: $path_2"

name_3="cr$7lv${lv}posx$8av$9"
path_3="$root_path/crcoeff_$7_lvcoeff_${lv}_posxcoeff_$8_avcoeff_$9/PPO_1"
echo "path 3: $path_3"

name_4="cr${10}lv${lv}posx${11}av${12}"
path_4="$root_path/crcoeff_${10}_lvcoeff_${lv}_posxcoeff_${11}_avcoeff_${12}/PPO_1"
echo "path 4: $path_4"

name_5="cr${13}lv${lv}posx${14}av${15}"
path_5="$root_path/crcoeff_${13}_lvcoeff_${lv}_posxcoeff_${14}_avcoeff_${15}/PPO_1"
echo "path 5: $path_5"

name_6="cr${16}lv${lv}posx${17}av${18}"
path_6="$root_path/crcoeff_${16}_lvcoeff_${lv}_posxcoeff_${17}_avcoeff_${18}/PPO_1"
echo "path 6: $path_6"

tensorboard --logdir_spec=$name_1:$path_1,$name_2:$path_2,$name_3:$path_3,$name_4:$path_4,$name_5:$path_5,$name_6:$path_6
