SCENE=holdout08
APPENDIX=_pure_sdfloss

python render_volume.py --mode validate_mesh --conf ./confs/wmask_face.conf --case ${SCENE} --appendix ${APPENDIX} --is_continue