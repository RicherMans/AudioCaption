import os
import argparse
import pandas as pd


def main(args):

    scp_file = args.scp_file
    eval_caption_file = args.eval_caption_file
    dev_output = args.dev_output
    eval_output = args.eval_output
    
    scp_filename = os.path.splitext(os.path.basename(scp_file))[0]

    if dev_output is None:
        dev_output = os.path.join(os.path.dirname(scp_file), scp_filename + "_dev.scp")

    if eval_output is None:
        eval_output = os.path.join(os.path.dirname(scp_file), scp_filename + "_eval.scp")

    eval_df = pd.read_json(eval_caption_file)
    eval_keys = eval_df["key"].apply(str).unique()

    with open(scp_file, "r") as f_read, open(dev_output, "w") as f_dev_write, open(eval_output, "w") as f_eval_write:
        for line in f_read.readlines():
            key = line.strip().split()[0]
            if key in eval_keys:
                f_eval_write.write(line)
            else:
                f_dev_write.write(line)

    print("Finish splitting {} to {} and {}".format(scp_file, dev_output, eval_output))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scp_file", type=str)
    parser.add_argument("eval_caption_file",type=str)
    parser.add_argument("--dev-output", type=str, default=None)
    parser.add_argument("--eval-output", type=str, default=None)

    args = parser.parse_args()
    main(args)
