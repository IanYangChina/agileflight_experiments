#!/bin/bash

# Pass number of rollouts as argument
if [ $1 ]
then
  N="$1"
else
  echo "Please specify the number of rollouts as the first variable"
  exit
fi

if [ $2 ]
then
  echo "Policy path received"
else
  echo "Please specify a policy path as the second variable"
  exit
fi

SUMMARY_FILE="evaluation.yaml"
"" > $SUMMARY_FILE

# Perform N evaluation runs
for i in $(eval echo {1..$N})
do
  # Publish simulator reset
  rostopic pub /kingfisher/dodgeros_pilot/off std_msgs/Empty "{}" --once
  rostopic pub /kingfisher/dodgeros_pilot/reset_sim std_msgs/Empty "{}" --once
  rostopic pub /kingfisher/dodgeros_pilot/enable std_msgs/Bool "data: true" --once
  rostopic pub /kingfisher/dodgeros_pilot/start std_msgs/Empty "{}" --once

  export ROLLOUT_NAME="rollout_""$i"
  echo "$ROLLOUT_NAME"

  cd ./envtest/ros/
  python3 evaluation_node.py &
  PY_PID="$!"

  python3 run_competition.py --ppo_path $2 &
  COMP_PID="$!"
  sleep 10.0

  cd -

  rostopic pub /kingfisher/start_navigation std_msgs/Empty "{}" --once

  # Wait until the evaluation script has finished
  while ps -p $PY_PID > /dev/null
  do
    sleep 1
  done

  cat "$SUMMARY_FILE" "./envtest/ros/summary.yaml" > "tmp.yaml"
  mv "tmp.yaml" "$SUMMARY_FILE"

  kill -SIGINT "$COMP_PID"
done

rm ./envtest/ros/summary.yaml
