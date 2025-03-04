# first argument should be the describer name, moondream if not specified
DESCRIBER=$1
if [ $# -eq 0 ]; then
    DESCRIBER="moondream"
fi
. settings.bash
TEST_MSG="$DESCRIBER virtual environment should install automatically."
python3 bvqa.py build -d $DESCRIBER
if [ $? -ne 0 ]; then
    echo "FAIL: $TEST_MSG"
fi
