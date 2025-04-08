From the terminal you can run individual parts. For example, to generate prompts:

```bash
python generate_prompts.py --num_prompts 100000 --output prompts.json --model meta-llama/Llama-2-70b --device 0
```

To deduplicate prompts:

```bash
python deduplicate_prompts.py --input prompts.json --output prompts_dedup.json
```

To generate webpages based on deduplicated prompts:

```bash
python generate_webpages.py --prompts prompts_dedup.json --output webpages.json --num_candidates 10 --model meta-llama/CodeLlama-2-7b-hf --device 0
```

Finally, to run the entire pipeline:

```bash
python main.py --num_prompts 100000 --num_candidates 10 --llama_model meta-llama/Llama-2-70b --code_llama_model meta-llama/CodeLlama-2-7b-hf --device 0
```

Added a version of submod_deduplication to archive which works only on prompts.
