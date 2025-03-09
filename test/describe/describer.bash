# first argument should be the describer name, moondream if not specified
DESCRIBER=$1
$MODEL=$2
if [ $# -eq 0 ]; then
    DESCRIBER="moondream"
fi
. settings.bash
TEST_MSG="$DESCRIBER description should finish without errors."
if [[ "$DESCRIBER" == *"qwen-vl"* ]]; then 
    "$BVQA_VENVS/$DESCRIBER/bin/python" bvqa.py describe -d $DESCRIBER -R test/data -r -t -m $MODEL -o
else
    "$BVQA_VENVS/$DESCRIBER/bin/python" bvqa.py describe -d $DESCRIBER -R test/data -r -t -m $MODEL
fi
if [ $? -ne 0 ]; then
    echo "FAIL: $TEST_MSG"
fi
