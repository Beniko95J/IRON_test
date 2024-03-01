SCENE=Luan_etal_2021/buddha

python render_surface.py --data_dir ./datasets/${SCENE}/train \
                                 --out_dir ./exp_iron_stage2/${SCENE} \
                                 --neus_ckpt_fpath ./exp_iron_stage1/${SCENE}/checkpoints/ckpt_100000.pth \
                                 --num_iters 50001 --gamma_pred