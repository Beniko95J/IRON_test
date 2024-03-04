SCENE=holdout08
APPENDIX=_test

python render_volume.py --mode train --conf ./confs/wmask_face.conf --case ${SCENE} --appendix ${APPENDIX}