
mkdir -p ./logs/hotpotqa/star/
mkdir -p ./logs/hotpotqa/chain/
mkdir -p ./logs/hotpotqa/tree/
mkdir -p ./logs/hotpotqa/fcg/

mkdir -p ./logs/multi_news/star/
mkdir -p ./logs/multi_news/chain/
mkdir -p ./logs/multi_news/tree/
mkdir -p ./logs/multi_news/fcg/

MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"

# HotpotQA + malicious document

# python main.py --mas star  --dataset hotpotqa --use_vllm_model --model $MODEL_NAME --attack none                        --defense none --num_samples 25 --log_file ./logs/hotpotqa/star/atttack_none.txt &
# python main.py --mas star  --dataset hotpotqa --use_vllm_model --model $MODEL_NAME --attack mal_doc --num_mal_clients 1 --defense none --num_samples 25 --log_file ./logs/hotpotqa/star/atttack_mal_doc_1.txt &
# python main.py --mas star  --dataset hotpotqa --use_vllm_model --model $MODEL_NAME --attack mal_doc --num_mal_clients 2 --defense none --num_samples 25 --log_file ./logs/hotpotqa/star/atttack_mal_doc_2.txt &
# python main.py --mas star  --dataset hotpotqa --use_vllm_model --model $MODEL_NAME --attack mal_doc --num_mal_clients 3 --defense none --num_samples 25 --log_file ./logs/hotpotqa/star/atttack_mal_doc_3.txt &

# python main.py --mas chain --dataset hotpotqa --use_vllm_model --model $MODEL_NAME --attack none                        --defense none --num_samples 25 --log_file ./logs/hotpotqa/chain/atttack_none.txt &
# python main.py --mas chain --dataset hotpotqa --use_vllm_model --model $MODEL_NAME --attack mal_doc --num_mal_clients 1 --defense none --num_samples 25 --log_file ./logs/hotpotqa/chain/atttack_mal_doc_1.txt &
# python main.py --mas chain --dataset hotpotqa --use_vllm_model --model $MODEL_NAME --attack mal_doc --num_mal_clients 2 --defense none --num_samples 25 --log_file ./logs/hotpotqa/chain/atttack_mal_doc_2.txt &
# python main.py --mas chain --dataset hotpotqa --use_vllm_model --model $MODEL_NAME --attack mal_doc --num_mal_clients 3 --defense none --num_samples 25 --log_file ./logs/hotpotqa/chain/atttack_mal_doc_3.txt &

# python main.py --mas tree  --dataset hotpotqa --use_vllm_model --model $MODEL_NAME --attack none                        --defense none --num_samples 25 --log_file ./logs/hotpotqa/tree/atttack_none.txt &
# python main.py --mas tree  --dataset hotpotqa --use_vllm_model --model $MODEL_NAME --attack mal_doc --num_mal_clients 1 --defense none --num_samples 25 --log_file ./logs/hotpotqa/tree/atttack_mal_doc_1.txt &
# python main.py --mas tree  --dataset hotpotqa --use_vllm_model --model $MODEL_NAME --attack mal_doc --num_mal_clients 2 --defense none --num_samples 25 --log_file ./logs/hotpotqa/tree/atttack_mal_doc_2.txt &
# python main.py --mas tree  --dataset hotpotqa --use_vllm_model --model $MODEL_NAME --attack mal_doc --num_mal_clients 3 --defense none --num_samples 25 --log_file ./logs/hotpotqa/tree/atttack_mal_doc_3.txt &

# python main.py --mas fcg   --dataset hotpotqa --use_vllm_model --model $MODEL_NAME --attack none                        --defense none --num_samples 25 --log_file ./logs/hotpotqa/fcg/atttack_none.txt &
# python main.py --mas fcg   --dataset hotpotqa --use_vllm_model --model $MODEL_NAME --attack mal_doc --num_mal_clients 1 --defense none --num_samples 25 --log_file ./logs/hotpotqa/fcg/atttack_mal_doc_1.txt &
# python main.py --mas fcg   --dataset hotpotqa --use_vllm_model --model $MODEL_NAME --attack mal_doc --num_mal_clients 2 --defense none --num_samples 25 --log_file ./logs/hotpotqa/fcg/atttack_mal_doc_2.txt &
# python main.py --mas fcg   --dataset hotpotqa --use_vllm_model --model $MODEL_NAME --attack mal_doc --num_mal_clients 3 --defense none --num_samples 25 --log_file ./logs/hotpotqa/fcg/atttack_mal_doc_3.txt &


# MultiNews + prompt injection 

python main.py --mas star  --dataset multi_news --use_vllm_model --model $MODEL_NAME --attack none                                 --defense none --num_samples 25 --log_file ./logs/multi_news/star/atttack_none.txt &
python main.py --mas star  --dataset multi_news --use_vllm_model --model $MODEL_NAME --attack prompt_injection --num_mal_clients 1 --defense none --num_samples 25 --log_file ./logs/multi_news/star/atttack_prompt_1.txt &
python main.py --mas star  --dataset multi_news --use_vllm_model --model $MODEL_NAME --attack prompt_injection --num_mal_clients 2 --defense none --num_samples 25 --log_file ./logs/multi_news/star/atttack_prompt_2.txt &
python main.py --mas star  --dataset multi_news --use_vllm_model --model $MODEL_NAME --attack prompt_injection --num_mal_clients 3 --defense none --num_samples 25 --log_file ./logs/multi_news/star/atttack_prompt_3.txt &

python main.py --mas chain --dataset multi_news --use_vllm_model --model $MODEL_NAME --attack none                                 --defense none --num_samples 25 --log_file ./logs/multi_news/chain/atttack_none.txt &
python main.py --mas chain --dataset multi_news --use_vllm_model --model $MODEL_NAME --attack prompt_injection --num_mal_clients 1 --defense none --num_samples 25 --log_file ./logs/multi_news/chain/atttack_prompt_1.txt &
python main.py --mas chain --dataset multi_news --use_vllm_model --model $MODEL_NAME --attack prompt_injection --num_mal_clients 2 --defense none --num_samples 25 --log_file ./logs/multi_news/chain/atttack_prompt_2.txt &
python main.py --mas chain --dataset multi_news --use_vllm_model --model $MODEL_NAME --attack prompt_injection --num_mal_clients 3 --defense none --num_samples 25 --log_file ./logs/multi_news/chain/atttack_prompt_3.txt &

python main.py --mas tree  --dataset multi_news --use_vllm_model --model $MODEL_NAME --attack none                                 --defense none --num_samples 25 --log_file ./logs/multi_news/tree/atttack_none.txt &
python main.py --mas tree  --dataset multi_news --use_vllm_model --model $MODEL_NAME --attack prompt_injection --num_mal_clients 1 --defense none --num_samples 25 --log_file ./logs/multi_news/tree/atttack_prompt_1.txt &
python main.py --mas tree  --dataset multi_news --use_vllm_model --model $MODEL_NAME --attack prompt_injection --num_mal_clients 2 --defense none --num_samples 25 --log_file ./logs/multi_news/tree/atttack_prompt_2.txt &
python main.py --mas tree  --dataset multi_news --use_vllm_model --model $MODEL_NAME --attack prompt_injection --num_mal_clients 3 --defense none --num_samples 25 --log_file ./logs/multi_news/tree/atttack_prompt_3.txt &

python main.py --mas fcg   --dataset multi_news --use_vllm_model --model $MODEL_NAME --attack none                                 --defense none --num_samples 25 --log_file ./logs/multi_news/fcg/atttack_none.txt &
python main.py --mas fcg   --dataset multi_news --use_vllm_model --model $MODEL_NAME --attack prompt_injection --num_mal_clients 1 --defense none --num_samples 25 --log_file ./logs/multi_news/fcg/atttack_prompt_1.txt &
python main.py --mas fcg   --dataset multi_news --use_vllm_model --model $MODEL_NAME --attack prompt_injection --num_mal_clients 2 --defense none --num_samples 25 --log_file ./logs/multi_news/fcg/atttack_prompt_2.txt &
python main.py --mas fcg   --dataset multi_news --use_vllm_model --model $MODEL_NAME --attack prompt_injection --num_mal_clients 3 --defense none --num_samples 25 --log_file ./logs/multi_news/fcg/atttack_prompt_3.txt &
