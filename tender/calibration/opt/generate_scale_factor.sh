#!/bin/bash
CHUNK_SIZE=256

# OPT - Linear & MHA
for nsamples in 128;do
  for seqlen in 2048;do
    for size in 30b;do
      for bits in 4;do
        case ${size} in 
          6.7b)
            if [ ${bits} = 4 ]; then
              decomp=8
            elif [ ${bits} = 8 ]; then
              decomp=4
            fi
          ;;
          13b)
            if [ ${bits} = 4 ]; then
              decomp=8
            elif [ ${bits} = 8 ]; then
              decomp=4
            fi
          ;;
          30b)
            if [ ${bits} = 4 ]; then
              decomp=10
            elif [ ${bits} = 8 ]; then
              decomp=8
            fi
          ;;
          66b)
            if [ ${bits} = 4 ]; then
              decomp=10
            elif [ ${bits} = 8 ]; then
              decomp=8
            fi
          ;;
        esac
        echo calibrating opt-${size} ${bits}bit chunk size of ${CHUNK_SIZE}
        echo linear + mha
        python run_calibration.py \
          --model-name "/data/models/opt-${size}" \
          --target "scale" \
          --output-path "/home/mskim/oaken-docker/oaken/quantizer/tender/opt_${size}_${bits}bit_${decomp}decomp_mha.pt" \
          --dataset-path "../../data/val.jsonl.zst" \
          --num-samples ${nsamples} \
          --seq-len ${seqlen} \
          --q_bits ${bits} \
          --decomp_factor ${decomp} \
          --chunk_size ${CHUNK_SIZE} \
          --quant_mha
      done
    done
  done
done

