import os.path as osp

import fingernet as fnet

if __name__ == '__main__':
    path = "../../datasets/NIST4"
    fnet.run_inference(
        input_path=osp.join(path, "orig"),
        output_path=osp.join(path, "fnet_out"),
        gpus=[1,2,3],
        batch_size=16,
        num_workers=4,
        recursive=False,
        mnt_degrees=True,
        compile_model=True
    )