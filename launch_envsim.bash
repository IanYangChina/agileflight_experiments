# Launch the simulator, unless it is already running
if [ -z $(pgrep visionsim_node) ]
then
  roslaunch envsim visionenv_sim.launch render:=True &
  ROS_PID="$!"
  echo $ROS_PID
  sleep 20
else
  ROS_PID=""
fi
