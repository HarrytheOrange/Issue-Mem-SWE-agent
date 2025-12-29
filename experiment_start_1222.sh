# !/usr/bin/env bash

sweagent run-batch --config config/KimiK2_Instruct0711_agenticMemSearch_1220_13w.yaml --agent.model.per_instance_cost_limit 10.00 --instances.type swe_bench --instances.subset verified --instances.split test --num_workers 20 --instances.slice :100 --instances.shuffle=False

sweagent run-batch --config config/Qwen3Coder30B_origin.yaml --agent.model.per_instance_cost_limit 1.00 --instances.type swe_bench --instances.subset verified --instances.split test --num_workers 20 --instances.slice :500 --instances.shuffle=False

sweagent run-batch --config config/dsv31t_agenticMemSearch_1224_31w.yaml --agent.model.per_instance_cost_limit 1.00 --instances.type swe_bench --instances.subset verified --instances.split test --num_workers 20 --instances.slice :100 --instances.shuffle=False

sweagent run-batch --config config/Qwen3Coder30B_agentic_rag_top3.yaml --agent.model.per_instance_cost_limit 1.00 --instances.type swe_bench --instances.subset verified --instances.split test --num_workers 20 --instances.slice :100 --instances.shuffle=False

sweagent run-batch --config config/Qwen3Coder30B_agenticMemSearch_1220_13w.yaml --agent.model.per_instance_cost_limit 1.00 --instances.type swe_bench --instances.subset verified --instances.split test --num_workers 20 --instances.slice :500 --instances.shuffle=False

sweagent run-batch --config config/Qwen3_235BInstruct2507_agenticMemSearch_1220_13w.yaml --agent.model.per_instance_cost_limit 1.00 --instances.type swe_bench --instances.subset verified --instances.split test --num_workers 20 --instances.slice :500 --instances.shuffle=False





sleep 300

sweagent run-batch --config config/claude_sonnet4_agenticMemSearch_1220_13w.yaml --agent.model.per_instance_cost_limit 20.00 --instances.type swe_bench --instances.subset verified --instances.split test --num_workers 10 --instances.slice 400:500 --instances.shuffle=False


sweagent run-batch --config config/dsv31t_agenticMemSearch_1220_13w.yaml --agent.model.per_instance_cost_limit 1.00 --instances.type swe_bench --instances.subset verified --instances.split test --num_workers 20 --instances.slice :500 --instances.shuffle=False

sweagent run-batch --config config/Gemini3Flash_agenticMemSearch_1220_13w.yaml --agent.model.per_instance_cost_limit 5.00 --instances.type swe_bench --instances.subset verified --instances.split test --num_workers 1 --instances.slice :500 --instances.shuffle=False

sweagent run-batch --config config/Qwen3_235BInstruct2507_origin.yaml --agent.model.per_instance_cost_limit 1.00 --instances.type swe_bench --instances.subset verified --instances.split test --num_workers 20 --instances.slice :500 --instances.shuffle=False

#cd ../SWE-bench

#python -m swebench.harness.run_evaluation --dataset_name princeton-nlp/SWE-bench_Verified --predictions_path /home/harry/Issue-Mem-SWE-agent/trajectories/harry/dsv31t_agenticMemSearch_1224_31w__deepseek--deepseek-v3-1-terminus__t-1.00__p-None__c-1.00___swe_bench_verified_test/preds.json --max_workers 20 --run_id validate-dsv31t-agenticMemSearch-1224-31w

#sleep 10

#python -m swebench.harness.run_evaluation --dataset_name princeton-nlp/SWE-bench_Verified --predictions_path /home/harry/Issue-Mem-SWE-agent/trajectories/harry/Qwen3Coder30B_agenticMemSearch_1224_31w__qwen--qwen3-coder-30b-a3b-instruct__t-1.00__p-None__c-1.00___swe_bench_verified_test/preds.json --max_workers 20 --run_id validate-qwen3coder30b-agenticMemSearch-1224-31w

#python -m swebench.harness.run_evaluation --dataset_name princeton-nlp/SWE-bench_Verified --predictions_path /home/harry/Issue-Mem-SWE-agent/trajectories/harry/Qwen3Coder30B_agenticMemSearch_1220_13w__qwen--qwen3-coder-30b-a3b-instruct__t-1.00__p-None__c-1.00___swe_bench_verified_test/preds.json --max_workers 20 --run_id validate-qwen3coder30b-agenticMemSearch-500-1220-13w

#python -m swebench.harness.run_evaluation --dataset_name princeton-nlp/SWE-bench_Verified --predictions_path /home/harry/Issue-Mem-SWE-agent/trajectories/harry/Qwen3Coder30B_agenticMemSearch_1220_13w__qwen--qwen3-coder-30b-a3b-instruct__t-1.00__p-None__c-1.00___swe_bench_verified_test/preds.json --max_workers 20 --run_id validate-qwen3coder30b-agenticMemSearch-1220-13w-500-2

#python -m swebench.harness.run_evaluation --dataset_name princeton-nlp/SWE-bench_Verified --predictions_path /home/harry/Issue-Mem-SWE-agent/trajectories/harry/dsv31t_agenticMemSearch_1220_13w__deepseek--deepseek-v3-1-terminus__t-1.00__p-None__c-1.00___swe_bench_verified_test/preds.json --max_workers 20 --run_id validate-dsv31t-agenticMemSearch-1220-13w-500

# python -m swebench.harness.run_evaluation --dataset_name princeton-nlp/SWE-bench_Verified --predictions_path /home/harry/Issue-Mem-SWE-agent/trajectories/harry/Qwen3Coder30B_origin__qwen--qwen3-coder-30b-a3b-instruct__t-1.00__p-None__c-1.00___swe_bench_verified_test/preds.json --max_workers 20 --run_id validate-qwen3coder30b-origin-500

# python -m swebench.harness.run_evaluation --dataset_name princeton-nlp/SWE-bench_Verified --predictions_path /home/harry/Issue-Mem-SWE-agent/trajectories/harry/KimiK2_Instruct0711_agenticMemSearch_1220_13w__moonshotai--kimi-k2__t-1.00__p-None__c-3.00___swe_bench_verified_test/preds.json --max_workers 20 --run_id validate-kimi-k2-agenticMemSearch-1220-13w-100

# python -m swebench.harness.run_evaluation --dataset_name princeton-nlp/SWE-bench_Verified --predictions_path /home/harry/Issue-Mem-SWE-agent/trajectories/harry/Gemini25flash_agenticMemSearch_1220_13w__google--gemini-2.5-flash__t-1.00__p-None__c-5.00___swe_bench_verified_test/preds.json --max_workers 20 --run_id validate-gemini25flash-agenticMemSearch-1220-13w-100