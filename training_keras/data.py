### Data class and associated helper methods

import numpy as np
import tables
import os
import time
from threading import Thread
import itertools
tables.set_blosc_max_threads(4)
import multiprocessing

class FilePreloader(Thread):
    def __init__(self, files_list, file_open, n_ahead=2):
        Thread.__init__(self)
        self.deamon = True
        self.n_concurrent = n_ahead
        self.files_list = files_list
        self.file_open = file_open
        self.loaded = {} ## a dict of the loaded objects
        self.should_stop = False
        
    def getFile(self, name):
        ## locks until the file is loaded, then return the handle
        return self.loaded.setdefault(name, self.file_open( name))

    def closeFile(self,name):
        ## close the file and
        if name in self.loaded:
            self.loaded.pop(name).close()
    
    def run(self):
        while not self.files_list:
            time.sleep(1)
        for name in itertools.cycle(self.files_list):
            if self.should_stop:
                break
            n_there = len(self.loaded.keys())
            if n_there< self.n_concurrent:
                print ("preloading",name,"with",n_there)
                self.getFile( name )
            else:
                time.sleep(5)

    def stop(self):
        print("Stopping FilePreloader")
        self.should_stop = True

def data_class_getter(name):
    """Returns the specified Data class"""
    data_dict = {
            "H5Data":H5Data,
            }
    try:
        return data_dict[name]
    except KeyError:
        print ("{0:s} is not a known Data class. Returning None...".format(name))
        return None


class Data(object):
    """Class providing an interface to the input training data.
        Derived classes should implement the load_data function.
        Attributes:
          file_names: list of data files to use for training
          batch_size: size of training batches
    """

    def __init__(self, batch_size, cache=None, spectators=False, weights=False):
        """Stores the batch size and the names of the data files to be read.
            Params:
              batch_size: batch size for training
        """
        self.batch_size = batch_size
        self.caching_directory = cache if cache else os.environ.get('GANINMEM','')
        self.spectators = spectators
        self.weights = weights
        self.fpl = None

    def set_caching_directory(self, cache):
        self.caching_directory = cache
        
    def set_file_names(self, file_names):
        ## hook to copy data in /dev/shm
        relocated = []
        if self.caching_directory:
            goes_to = self.caching_directory
            goes_to += str(os.getpid())
            os.system('mkdir %s '%goes_to)
            os.system('rm %s/* -f'%goes_to) ## clean first if anything
            for fn in file_names:
                relocate = goes_to+'/'+fn.split('/')[-1]
                if not os.path.isfile( relocate ):
                    print ("copying %s to %s"%( fn , relocate))
                    if os.system('cp %s %s'%( fn ,relocate))==0:
                        relocated.append( relocate )
                    else:
                        print ("was enable to copy the file",fn,"to",relocate)
                        relocated.append( fn ) ## use the initial one
                else:
                    relocated.append( relocate )
                        
            self.file_names = relocated
        else:
            self.file_names = file_names
            
        if self.fpl:
            self.fpl.files_list = self.file_names

    def inf_generate_data(self):
        """Yields batches of training data forever."""
        while True:
            try:
                for B in self.generate_data():
                    yield B
            except StopIteration as si:
                print ("start over generator loop")

    def generate_data(self):
       """Yields batches of training data until none are left."""
       leftovers = None
       for cur_file_name in self.file_names:
           if self.spectators and self.weights:
               cur_file_features, cur_file_labels, cur_file_spectators, cur_file_weights = self.load_data(cur_file_name)
           elif self.weights:
               cur_file_features, cur_file_labels, cur_file_weights = self.load_data(cur_file_name)
           elif self.spectators: 
               cur_file_features, cur_file_labels, cur_file_spectators = self.load_data(cur_file_name)               
           else:
               cur_file_features, cur_file_labels = self.load_data(cur_file_name)
           # concatenate any leftover data from the previous file
           if leftovers is not None:
               cur_file_features = self.concat_data( leftovers[0], cur_file_features )
               cur_file_labels = self.concat_data( leftovers[1], cur_file_labels )
               if self.spectators:
                   cur_file_spectators = self.concat_data( leftovers[2], cur_file_spectators)                   
               leftovers = None
           num_in_file = self.get_num_samples( cur_file_features )
           for cur_pos in range(0, num_in_file, self.batch_size):
               next_pos = cur_pos + self.batch_size 
               if next_pos <= num_in_file:
                   if self.spectators and self.weights:
                       yield ( self.get_batch( cur_file_features, cur_pos, next_pos, expand_dims = True , squeeze = True),
                               self.get_batch( cur_file_labels, cur_pos, next_pos, squeeze = True ),
                               self.get_batch( cur_file_spectators, cur_pos, next_pos, expand_dims = True, squeeze = True ),
                               self.get_batch( cur_file_weights, cur_pos, next_pos, prod = True ) )
                   elif self.weights:
                       yield ( self.get_batch( cur_file_features, cur_pos, next_pos, expand_dims = True , squeeze = True),
                               self.get_batch( cur_file_labels, cur_pos, next_pos, squeeze = True ),
                               self.get_batch( cur_file_weights, cur_pos, next_pos, prod = True ) )
                   elif self.spectators:
                       yield ( self.get_batch( cur_file_features, cur_pos, next_pos, expand_dims = True , squeeze = True),
                               self.get_batch( cur_file_labels, cur_pos, next_pos, squeeze = True ),
                               self.get_batch( cur_file_spectators, cur_pos, next_pos, expand_dims = True ) )
                   else:
                       yield ( self.get_batch( cur_file_features, cur_pos, next_pos, expand_dims = True , squeeze = True),
                               self.get_batch( cur_file_labels, cur_pos, next_pos, squeeze = True) )
               else:
                   if self.spectators and self.weights:
                       leftovers = ( self.get_batch( cur_file_features, cur_pos, num_in_file, expand_dims = True, squeeze = True ),
                                     self.get_batch( cur_file_labels, cur_pos, num_in_file, squeeze = True ),
                                     self.get_batch( cur_file_spectators, cur_pos, num_in_file, expand_dims = True , squeeze = True),
                                     self.get_batch( cur_file_weights, cur_pos, num_in_file, prod = True ) )
                   elif self.weights:
                       leftovers = ( self.get_batch( cur_file_features, cur_pos, num_in_file, expand_dims = True , squeeze = True),
                                     self.get_batch( cur_file_labels, cur_pos, num_in_file, squeeze = True ),
                                     self.get_batch( cur_file_weights, cur_pos, num_in_file, prod = True) )
                   elif self.spectators:
                       leftovers = ( self.get_batch( cur_file_features, cur_pos, num_in_file, expand_dims = True , squeeze = True),
                                     self.get_batch( cur_file_labels, cur_pos, num_in_file, squeeze = True ),
                                     self.get_batch( cur_file_spectators, cur_pos, num_in_file, expand_dims = True , squeeze = True) )
                   else:
                       leftovers = ( self.get_batch( cur_file_features, cur_pos, num_in_file, expand_dims = True , squeeze = True),
                                     self.get_batch( cur_file_labels, cur_pos, num_in_file, squeeze = True ) )

    def count_data(self):
        """Counts the number of data points across all files"""
        num_data = 0
        for cur_file_name in self.file_names:
           if self.spectators and self.weights:
               cur_file_features, cur_file_labels, cur_file_spectators, cur_file_weights = self.load_data(cur_file_name)
           elif self.weights:
               cur_file_features, cur_file_labels, cur_file_weights = self.load_data(cur_file_name)
           elif self.spectators:
               cur_file_features, cur_file_labels, cur_file_spectators = self.load_data(cur_file_name)
           else:
               cur_file_features, cur_file_labels = self.load_data(cur_file_name)
           num_data += self.get_num_samples( cur_file_labels )
        return num_data

    def is_numpy_array(self, data_array):
        return (isinstance( data_array, np.ndarray ) or isinstance( data_array, tables.CArray ) )

    def get_batch(self, data_array, start_pos, end_pos, expand_dims=False, squeeze=False, prod=False):
        """Input: a numpy array or list of numpy arrays.
            Gets elements between start_pos and end_pos in each array"""
        if type(data_array) == list:
            lout = []
            for arr in data_array:
                a = arr[start_pos:end_pos]
                if expand_dims:
                    a = np.reshape(a,a.shape+(1,))
                lout.append(a)
            out = np.stack(lout,axis=-1)
            if squeeze: 
                out = np.squeeze(out)
            elif prod:
                out = np.prod(out,axis=-1)
            return out
        else:
            return data_array[start_pos:end_pos] 

    def concat_data(self, data1, data2):
        """Input: data1 as numpy array or list of numpy arrays.  data2 in the same format.
           Returns: numpy array or list of arrays, in which each array in data1 has been
             concatenated with the corresponding array in data2"""
        if type(data1)==list:
            return [ self.concat_data( d1, d2 ) for d1,d2 in zip(data1,data2) ]
        else:
            return np.concatenate( (data1, data2) )


    def get_num_samples(self, data_array):
        """Input: dataset consisting of a numpy array or list of numpy arrays.
            Output: number of samples in the dataset"""
        if type(data_array)==list:
            return data_array[0].shape[0]
        else:
            return data_array.shape[0]

    def load_data(self, in_file):
        """Input: name of file from which the data should be loaded
            Returns: tuple (X,Y) where X and Y are numpy arrays containing features 
                and labels, respectively, for all data in the file
            Not implemented in base class; derived classes should implement this function"""
        raise NotImplementedError

class H5Data(Data):
    """Loads data stored in hdf5 files
        Attributes:
          features_name, labels_name, spectators_name, weights_name: 
          names of the datasets containing the features, 
          labels, spectators, and weights respectively
    """
    def __init__(self, batch_size,
                 cache=None,
                 preloading=0,
                 features_name=['features'],
                 labels_name=['labels'],
                 spectators_name = None,
                 weights_name = None):
        """Initializes and stores names of feature and label datasets"""
        super(H5Data, self).__init__(batch_size,cache,(spectators_name is not None),(weights_name is not None))
        self.features_name = features_name
        self.labels_name = labels_name        
        self.spectators_name = spectators_name
        self.weights_name = weights_name
        ## initialize the data-preloader
        self.fpl = None
        if preloading:
            self.fpl = FilePreloader( [] , file_open = lambda n : tables.open_file(n,'r'), n_ahead=preloading)
            self.fpl.start()          
       

    def load_data(self, in_file_name):
        """Loads numpy arrays from H5 file.
            If the features/labels groups contain more than one dataset,
            we load them all, alphabetically by key."""
        if self.fpl:
            h5_file = self.fpl.getFile( in_file_name )
        else:
            h5_file = tables.open_file( in_file_name, 'r' )
        X = self.load_hdf5_data( h5_file, self.features_name)
        Y = self.load_hdf5_data( h5_file, self.labels_name)
        if self.spectators_name is not None:
            Z = self.load_hdf5_data( h5_file, self.spectators_name)
        if self.weights_name is not None:
            W = self.load_hdf5_data( h5_file, self.weights_name)
        #if self.fpl:
        #    self.fpl.closeFile( in_file_name )
        #else:
        #    h5_file.close()
        if self.spectators_name is not None and self.weights_name is not None:
            return X,Y,Z,W
        elif self.spectators_name is not None:
            return X,Y,Z
        elif self.weights_name is not None:
            return X,Y,W
        else:
            return X,Y

    def load_hdf5_data(self, data_file, keys):
        """Returns a CArray or (possibly nested) list of CArrays 
            corresponding to the group structure of the input HDF5 data."""
        out = []
        for key in keys:
            a = getattr(data_file.root,key)
            out.append(a)
        return out

    def count_data(self):
        """This is faster than using the parent count_data
            because the datasets do not have to be loaded
            as numpy arrays"""
        num_data = 0
        for in_file_name in self.file_names:
            with tables.open_file(in_file_name,'r') as f:
                num_data += getattr(f.root, self.labels_name[0]).shape[0]
        return num_data

    def finalize(self):
        if self.fpl:
            self.fpl.stop()
