# run base.bash
BVQA_ROOT_REL="$(dirname "$0")/../.."
export BVQA_ROOT=$(realpath "$BVQA_ROOT_REL")
export BVQA_VENVS="$BVQA_ROOT/venvs"
cd $BVQA_ROOT
