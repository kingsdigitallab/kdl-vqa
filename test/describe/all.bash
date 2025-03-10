cd "$(dirname "$0")"
bash describer.bash moondream vikhyatk/moondream2
CUDA_VISIBLE_DEVICES=0 bash describer.bash qwen-vl Qwen/Qwen2.5-VL-3B-Instruct
bash describer.bash smol HuggingFaceTB/SmolVLM-Instruct
# TODO: understand why it hangs on 'object' qst for susan-q image
# bash describer.bash ollama llama3.2-vision
bash describer.bash ollama minicpm-v
bash describer.bash ovis AIDC-AI/Ovis2-1B
python3 bvqa.py report -R test/data -t
