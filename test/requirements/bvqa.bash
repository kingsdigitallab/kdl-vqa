. settings.bash
TEST_MSG="bvqa --help doesn't need third-party package."
python3 bvqa.py --help
if [ $? -ne 0 ]; then
    echo "FAIL: $TEST_MSG"
fi
