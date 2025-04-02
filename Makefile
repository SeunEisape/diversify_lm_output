# Prompt of interest
PROMPT_OF_INTEREST = haiku

all_prompts:
	python /home/sanjayss/gpu_scheduler/reserve.py -- python /home/eisape/projects/diversify_lm_output/utils/olmo_inference/run_all_normal_prompts.py --num_completions 200 --max_tokens 500
	python /home/sanjayss/gpu_scheduler/reserve.py -- python /home/eisape/projects/diversify_lm_output/utils/olmo_inference/run_all_random_prompts.py --num_completions 200 --max_tokens 500

normal_prompts:
	gpu python utils/olmo_inference/run_all_normal_prompts.py --num_completions 200 --max_tokens 500

random_prompts:
	gpu python utils/olmo_inference/run_all_random_prompts.py --num_completions 200 --max_tokens 500

mean_and_std:
	python utils/eval/mean_and_std.py --jsonl_file completions_eval_store/${PROMPT_OF_INTEREST}/${PROMPT_OF_INTEREST}_normal_prompt_output.jsonl
	python utils/eval/mean_and_std.py --jsonl_file completions_eval_store/${PROMPT_OF_INTEREST}/${PROMPT_OF_INTEREST}_random_prompt_output.jsonl

ngram_entropy:
	python utils/eval/ngram_entropy.py --jsonl_file completions_eval_store/${PROMPT_OF_INTEREST}/${PROMPT_OF_INTEREST}_normal_prompt_output.jsonl
	python utils/eval/ngram_entropy.py --jsonl_file completions_eval_store/${PROMPT_OF_INTEREST}/${PROMPT_OF_INTEREST}_random_prompt_output.jsonl
