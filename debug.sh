set -e

debug_1_name="train-parquet-debug"
#debug_1_name="test-parquet-debug"
python -u debug.py > logs/$debug_1_name.out 2> logs/$debug_1_name.err &
