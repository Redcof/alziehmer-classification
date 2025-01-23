import os
import re
import tarfile
import zipfile


def _is_tarxz(filename):
    return filename.endswith(".tar.xz")


def _is_tar(filename):
    return filename.endswith(".tar")


def _is_targz(filename):
    return filename.endswith(".tar.gz")


def _is_tgz(filename):
    return filename.endswith(".tgz")


def _is_gzip(filename):
    return filename.endswith(".gz") and not filename.endswith(".tar.gz")


def _is_zip(filename):
    return filename.endswith(".zip")


def extract_archive(from_path, to_path=None, remove_finished=False, dry_run=False):
    print("Extracting archive " + from_path + " to " + to_path)
    if dry_run:
        print("This is a dry run, skipping...")
        return
    if to_path is None:
        to_path = os.path.dirname(from_path)
    
    if _is_tar(from_path):
        with tarfile.open(from_path, 'r') as tar:
            tar.extractall(path=to_path)
    elif _is_targz(from_path) or _is_tgz(from_path):
        with tarfile.open(from_path, 'r:gz') as tar:
            tar.extractall(path=to_path)
    elif _is_tarxz(from_path):
        with tarfile.open(from_path, 'r:xz') as tar:
            tar.extractall(path=to_path)
    elif _is_gzip(from_path):
        to_path = os.path.join(to_path, os.path.splitext(
            os.path.basename(from_path))[0])
        with open(to_path, "wb") as out_f, gzip.GzipFile(from_path) as zip_f:
            out_f.write(zip_f.read())
    elif _is_zip(from_path):
        with zipfile.ZipFile(from_path, 'r') as z:
            z.extractall(to_path)
    else:
        raise ValueError("Extraction of {} not supported".format(from_path))
    
    if remove_finished:
        os.remove(from_path)


URL_REGEX = re.compile(
    r'^(?:http|ftp)s?://'  # http:// or https://
    # domain...
    r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'
    r'localhost|'  # localhost...
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
    r'(?::\d+)?'  # optional port
    r'(?:/?|[/?]\S+)$', re.IGNORECASE)


def is_url(url):
    return re.match(URL_REGEX, url) is not None


def is_kaggle_url(url):
    prefixes = ['kaggle.com',
                'www.kaggle.com',
                'http://kaggle.com',
                'http://www.kaggle.com',
                'https://www.kaggle.com',
                'https://kaggle.com']
    return any((url.startswith(prefix) for prefix in prefixes))


def get_kaggle_dataset_id(dataset_id_or_url):
    parts = []
    dataset_id_or_url = dataset_id_or_url.replace("/datasets/", "/")
    if is_kaggle_url(dataset_id_or_url):
        parts = dataset_id_or_url.split('?')[0].split(
            'kaggle.com/')[1].split('/')[:2]
    elif not is_url(dataset_id_or_url):
        parts = dataset_id_or_url.split('/')[:2]
    assert len(parts) == 2, 'Invalid Kaggle dataset URL or ID: ' + \
                            dataset_id_or_url
    return '/'.join(parts)


def download_kaggle_dataset(dataset_url, data_dir, force=False, dry_run=False, verify_ssl=True):
    dataset_id = get_kaggle_dataset_id(dataset_url)
    id = dataset_id.split('/')[1]
    target_dir = os.path.join(data_dir, id)
    
    if not force and os.path.exists(target_dir) and len(os.listdir(target_dir)) > 0:
        print('Skipping, found downloaded files in "{}" (use force=True to force download)'.format(
            target_dir))
        return
    
    if not dry_run:
        from kaggle import api
        from kaggle import rest
        api.api_client.configuration.verify_ssl = verify_ssl
        api.api_client.rest_client = rest.RESTClientObject(api.api_client.configuration)
        api.authenticate()
        if dataset_id.split('/')[0] == 'competitions' or dataset_id.split('/')[0] == 'c':
            api.competition_download_files(
                id,
                target_dir,
                force=force,
                quiet=False)
            zip_fname = target_dir + '/' + id + '.zip'
            extract_archive(zip_fname, target_dir)
            try:
                os.remove(zip_fname)
            except OSError as e:
                print('Could not delete zip file, got' + str(e))
        else:
            api.dataset_download_files(
                dataset_id,
                target_dir,
                force=force,
                quiet=False,
                unzip=True)
    
    else:
        print("This is a dry run, skipping..")
