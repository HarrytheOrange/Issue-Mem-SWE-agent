# !/usr/bin/env bash

sweagent run-batch --config config/dsv31t_origin.yaml --agent.model.per_instance_cost_limit 1.00 --instances.type swe_bench --instances.subset verified --instances.split test --num_workers 20 --instances.slice :100 --instances.shuffle=False

sweagent run-batch --config config/dsv31t_agenticMemSearch_1219_2.yaml --agent.model.per_instance_cost_limit 1.00 --instances.type swe_bench --instances.subset verified --instances.split test --num_workers 20 --instances.slice :100 --instances.shuffle=False

sweagent run-batch --config config/dsv31t_agenticMemSearch_1219_2.yaml --agent.model.per_instance_cost_limit 1.00 --instances.type swe_bench --instances.subset verified --instances.split test --num_workers 20 --instances.slice :100 --instances.shuffle=False
#sleep 300

#sweagent run-batch --config config/dsv31t_agenticMemSearch_1219_1.yaml --agent.model.per_instance_cost_limit 1.00 --instances.type swe_bench --instances.subset verified --instances.split test --num_workers 20 --instances.slice :100 --instances.shuffle=False

#cd ../SWE-bench

#python -m swebench.harness.run_evaluation --dataset_name princeton-nlp/SWE-bench_Verified --predictions_path /home/harry/Issue-Mem-SWE-agent/trajectories/harry/dsv31t_agenticMemSearch_1219_2__deepseek--deepseek-v3-1-terminus__t-1.00__p-None__c-1.00___swe_bench_verified_test/preds.json --max_workers 20 --run_id validate-dsv31t-agentmemsearch-1219-100-2

#sleep 10

#python -m swebench.harness.run_evaluation --dataset_name princeton-nlp/SWE-bench_Verified --predictions_path /home/harry/Issue-Mem-SWE-agent/trajectories/harry/dsv31t_origin__deepseek--deepseek-v3-1-terminus__t-1.00__p-None__c-1.00___swe_bench_verified_test/preds.json --max_workers 20 --run_id validate-dsv31t-origin-100-2
