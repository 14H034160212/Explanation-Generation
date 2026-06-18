#!/bin/bash
# Orchestrator: when each Qwen3 chain finishes, launch the matching LLaMA-2 chain on the freed GPU.
cd /data/qbao775/Explanation-Generation
log(){ echo "[orch $(date '+%H:%M:%S')] $*"; }

# --- Cardiff: wait for Qwen3 Cardiff (GPU 6) -> launch LLaMA-2 Cardiff on GPU 6 ---
log "waiting for Qwen3 Cardiff to finish (frees GPU 6)..."
while [ ! -f .tierC_qwen3_cardiff_DONE ]; do sleep 120; done
log "Qwen3 Cardiff done. Launching LLaMA-2 Cardiff on GPU 6."
sed 's/__GPU__/6/' run_tierC_llama2_cardiff.sh > run_tierC_llama2_cardiff_gpu.sh
rm -f .tierC_llama2_cardiff_DONE
setsid bash run_tierC_llama2_cardiff_gpu.sh > rl_tierC_llama2_cardiff.log 2>&1 < /dev/null &

# --- Sydney: wait for Qwen3 Sydney (GPU 0) -> launch LLaMA-2 Sydney on GPU 0 ---
log "waiting for Qwen3 Sydney to finish (frees GPU 0)..."
while [ ! -f .tierC_qwen3_sydney_DONE ]; do sleep 120; done
log "Qwen3 Sydney done. Launching LLaMA-2 Sydney on GPU 0."
sed 's/__GPU__/0/' run_tierC_llama2_sydney.sh > run_tierC_llama2_sydney_gpu.sh
rm -f .tierC_llama2_sydney_DONE
setsid bash run_tierC_llama2_sydney_gpu.sh > rl_tierC_llama2_sydney.log 2>&1 < /dev/null &

# --- wait for both LLaMA-2 chains ---
log "waiting for both LLaMA-2 chains to finish..."
while [ ! -f .tierC_llama2_cardiff_DONE ] || [ ! -f .tierC_llama2_sydney_DONE ]; do sleep 120; done
touch /data/qbao775/Explanation-Generation/.tierC_ALL_DONE
log "ALL Tier-C configs complete."
