import urllib.request
import sys
import os
import tarfile

class data_loader():
    def __init__(self):
        return
    
    def dataload(self):
        if os.path.exists('data') == False:
            self.create_folder()
            self.file_download()
            self.open_data()
        elif os.path.exists('data') & os.path.exists('data/simple-examples.tgz') == False:
            self.file_download()
            self.open_data()
        elif os.path.exists('data') & os.path.exists('data/simple-examples.tgz') & os.path.exists('data/simple-examples/data') == False:
            self.open_data()   
        
    def create_folder(self):
        os.mkdir('data')
        
    def file_download(self):
        url = 'http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz'
        name = 'data/simple-examples.tgz'
        urllib.request.urlretrieve(url, name)
        
    def open_data(self):
        with tarfile.open('data/simple-examples.tgz', 'r:gz') as tf:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tf, "data")