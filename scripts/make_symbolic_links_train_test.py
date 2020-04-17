'''                                                                                                                              
Script for making symbolic links for train/test

@author: jmduarte
'''

from __future__ import print_function
import glob
import argparse
import os
import sys
import random

def get_num_events(filepath, treename, selection=None):
    import ROOT as rt
    rt.gROOT.SetBatch(True)
    import traceback
    try:
        f = rt.TFile.Open(filepath)
        tree = f.Get(str(treename))
        if not tree:
            raise RuntimeError('Cannot find tree %s in file %s' % (treename, filepath))
        if selection is None:
            return tree.GetEntries()
        else:
            return tree.GetEntries(selection)
    except:
        print('Error reading %s:\n%s' % (filepath, traceback.format_exc()))
        return None


def main():
    parser = argparse.ArgumentParser('Make symbolic links for training/testing ntuples')
    parser.add_argument('inputdir',
                        help='Input diretory.'
                    )
    parser.add_argument('-t', '--testfrac',
                        default=0.15, type=float,
                        help='Test fraction'
                    )
    args = parser.parse_args()
    
    first_level = glob.glob(args.inputdir+'/*')
    
    args.inputdir = os.path.realpath(args.inputdir)
    traindir = os.path.join(args.inputdir,'train')
    testdir = os.path.join(args.inputdir,'test')

    input_files = {}
    input_files_test = {}
    input_files_train = {}
    num_events = {}
    num_events_test = {}
    num_events_train = {}
    counter = 0
    for sample_dir in first_level:
        input_files[sample_dir] = []
        input_files_test[sample_dir] = []
        input_files_train[sample_dir] = []
        num_events[sample_dir] = []
        num_events_test[sample_dir] = []
        num_events_train[sample_dir] = []
        for dp, dn, filenames in os.walk(sample_dir, followlinks=True):
            if 'failed' in dp or 'ignore' in dp:
                continue
            for f in filenames:
                if not f.endswith('.root'):
                    continue
                fullpath = os.path.realpath(os.path.join(dp, f))
                nevts = get_num_events(fullpath, 'deepntuplizer/tree')
                if nevts:
                    input_files[sample_dir].append(fullpath)
                    num_events[sample_dir].append(nevts)
                    counter += 1
                    if counter%100==0:
                        print('%d files processed...' % counter)
                else:
                    print('Ignore erroneous file %s' % fullpath)

        total_files = len(input_files[sample_dir])
        test_files = max(int(args.testfrac*total_files),1)
        train_files = total_files - test_files 

        inds = set(random.sample(list(range(total_files)), test_files))

        input_files_test[sample_dir] = [n for i,n in enumerate(input_files[sample_dir]) if i in inds]
        input_files_train[sample_dir] = [n for i,n in enumerate(input_files[sample_dir]) if i not in inds]

        for f in input_files_test[sample_dir]:
            link = f.replace(args.inputdir,testdir)
            folder = os.path.dirname(link)
            if f!=link:
                os.makedirs(folder,exist_ok=True)
                os.symlink(f,link)
            else:
                print("ERROR: no link made for %s!"%f)

        for f in input_files_train[sample_dir]:
            link = f.replace(args.inputdir,traindir)
            folder = os.path.dirname(link)
            if f!=link:
                os.makedirs(folder,exist_ok=True)
                os.symlink(f,link)
            else:
                print("ERROR: no link made for %s!"%f)
            
        print("SAMPLE:", sample_dir)
        print('total files:', total_files)
        print('test files:', test_files)
        print('train files:', train_files)

if __name__ == '__main__':
    main()


