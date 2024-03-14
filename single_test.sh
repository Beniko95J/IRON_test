SCENE=holdout04

python render_surface.py --data_dir ./datasets/synthetic_face_sparse/${SCENE}/test \
                                 --out_dir ./exp_face_stage2/${SCENE} \
                                 --neus_ckpt_fpath ./exp_face_stage1/${SCENE}/checkpoints/ckpt_100000.pth \
                                 --num_iters 50001 --gamma_pred --render_all