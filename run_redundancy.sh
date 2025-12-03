
mkdir -p ./logs/hotpotqa/star/
mkdir -p ./logs/hotpotqa/chain/
mkdir -p ./logs/hotpotqa/tree/
mkdir -p ./logs/hotpotqa/fcg/

mkdir -p ./logs/multi_news/star/
mkdir -p ./logs/multi_news/chain/
mkdir -p ./logs/multi_news/tree/
mkdir -p ./logs/multi_news/fcg/

MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"

Attack: prompt_injection, num_mal_clients=1, Defense: redundancy
python main_redundancy.py \
    --mas star \
    --dataset multi_news \
    --use_vllm_model \
    --model $MODEL_NAME \
    --attack prompt_injection \
    --num_mal_clients 1 \
    --defense redundancy \
    --redundancy_count 3 \
    --voting_strategy semantic \
    --similarity_threshold 0.7 \
    --num_samples 25 \
    --log_file ./logs/multi_news/star/attack_prompt_1_redundancy.txt


python main_redundancy.py \
    --mas star \
    --dataset multi_news \
    --use_vllm_model \
    --model $MODEL_NAME \
    --attack prompt_injection \
    --num_mal_clients 2 \
    --defense redundancy \
    --redundancy_count 3 \
    --voting_strategy semantic \
    --similarity_threshold 0.7 \
    --num_samples 25 \
    --log_file ./logs/multi_news/star/attack_prompt_2_redundancy.txt


python main_redundancy.py \
    --mas star \
    --dataset multi_news \
    --use_vllm_model \
    --model $MODEL_NAME \
    --attack prompt_injection \
    --num_mal_clients 3 \
    --defense redundancy \
    --redundancy_count 3 \
    --voting_strategy semantic \
    --similarity_threshold 0.7 \
    --num_samples 25 \
    --log_file ./logs/multi_news/star/attack_prompt_3_redundancy.txt


