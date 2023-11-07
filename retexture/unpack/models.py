import argparse
import shutil
import zipfile
import os
import os.path as osp

def parse_args():
    parser = argparse.ArgumentParser(description="Process the --path argument.")
    parser.add_argument('--path', type=str, help='Path to the input directory')
    parser.add_argument('--out', type=str, help='Path to the output directory')
    

    args = parser.parse_args()
    return args


def safe_mkdir(path):
    # Create output directory if it does not exist
    if not os.path.exists(path):
        os.makedirs(path)
 

def unzip_file(zip_path, output_path):
    """
    Unzips a zip file to a specified output path.

    Parameters:
    zip_path (str): The path to the .zip file to be extracted.
    output_path (str): The destination path where the contents will be extracted.
    """

    safe_mkdir(output_path)

    # Extract the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_path)

    print(f"Extracted {zip_path} to {output_path}")



def dae2basename(file_path):
    """
    Renames/moves a .dae file to have the name of the base directory it is in.
    
    Parameters:
    file_path (str): The path to the .dae file.
    """

    dae = [x for x in os.listdir(file_path) if '.dae' in x][0]
    dae = osp.join(file_path,dae)

    # Get the directory name and the base name of that directory
    dir_name = os.path.dirname(dae)
    base_name = os.path.basename(dir_name)

    if not base_name:
        raise ValueError("The file is located at the root directory, cannot rename to directory base name.")

    outpath = os.path.join(osp.dirname(dir_name), f"{base_name}.dae")
    shutil.move(dae, outpath)


def remove_folders_in_path(clean_path):
    """
    Removes all folders in the specified path without removing files.

    Parameters:
    clean_path (str): The path where folders will be removed.
    """

    for item in os.listdir(clean_path):
        item_path = os.path.join(clean_path, item)

        if os.path.isdir(item_path):
            shutil.rmtree(item_path)

def main():
    
    args = parse_args()
    
    list_fullpath = lambda parent: [osp.join(parent,x) for x in os.listdir(parent)]
    categories = list_fullpath(args.path)

    list_concat = lambda x: sum([list_fullpath(y) for y in x],[])
    models = list_concat(categories)
    zips = [x for x in list_concat(models) if '.zip' in x]

    zips = {osp.basename(osp.dirname(z)):z for z in zips}
    print(zips.keys())

    safe_mkdir(args.out)
    for k,v in zips.items():
        outdir = osp.join(args.out,k)
        unzip_file(v, outdir)
        dae2basename(outdir)

    remove_folders_in_path(args.out)

if __name__ == "__main__":
    main()
