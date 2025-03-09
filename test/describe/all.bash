cd "$(dirname "$0")"
bash describer.bash moondream
bash describer.bash qwen-vl
bash describer.bash smol
bash describer.bash ollama
python3 bvqa.py report -R test/data -t
