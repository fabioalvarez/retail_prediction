from pathlib import Path

def get_file_folders(s3_client, Bucket, Prefix):
    # Initiate variables
    folders = list()
    files = list()
    # Define 
    default_kwargs = {
        "Bucket": Bucket,
        "Prefix": Prefix
    }
    # Get metadata of objects in S3 bucket
    response = s3_client.list_objects_v2(**default_kwargs)
    contents = response.get('Contents')

    # Loop over the list of objects in S3
    for result in contents:
        key = result.get('Key')
        if key[-1] == '/':
            folders.append(key)
        else:
            files.append(key)

    return folders, files

def download_files(s3_client, bucket_name:str, local_path:str, file_names:list, folders:list, prefix:str):

    local_path = Path(local_path)

    for folder in folders:
        folder_path = Path.joinpath(local_path, folder)
        folder_path.mkdir(parents=True, exist_ok=True)

    for file_name in file_names:
        file_path = file_name.replace(prefix+'/', '')
        file_path = Path.joinpath(local_path, file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        s3_client.download_file(
            bucket_name,
            file_name,
            str(file_path)
        )