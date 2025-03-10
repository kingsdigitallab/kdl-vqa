cd "$(dirname "$0")"
bash describer.bash moondream 
CUDA_VISIBLE_DEVICES=1 bash describer.bash qwen-vl Qwen/Qwen2.5-VL-3B-Instruct
bash describer.bash smol HuggingFaceTB/SmolVLM-Instruct
bash describer.bash ollama llama3.2-vision
python3 bvqa.py report -R test/data -t
