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
            tf.extractall('data') 