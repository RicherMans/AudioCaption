import sys
import argparse
from pathlib import Path
import h5py

sys.path.append(str(Path.cwd()))
import utils.kaldi_io as kaldi_io

def main(copy_stream, write_hdf5):
    write_scp = write_hdf5.split(".")[0] + ".scp"
    with h5py.File(write_hdf5, "w") as feature_store, open(write_scp, "w") as key_store:
        for key, feature in kaldi_io.read_mat_ark(copy_stream):
            feature_store[key] = feature
            key_store.write(key + "\n")
    print("Successfully copy feature to {}".format(write_hdf5))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("copy_ark", type=str)
    parser.add_argument("write_hdf5", type=str)

    args = parser.parse_args()
    
    assert Path(args.write_hdf5).parent.exists(), "Directory for desitination hdf5 file does not exist, please create it first"

    main(args.copy_ark, args.write_hdf5)
