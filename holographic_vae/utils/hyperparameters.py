

from argparse import ArgumentError
import os
import json
import hashlib
import numpy as np

from typing import *

DICT_HASH_FILEPATH = ''

def get_list_of_combos_from_grid(grid):
    '''
    Takes as input a Dict where values are lists of possible combinations (e.g. a grid for hyperparameter search).
    Returns a List of Dicts, each containing the same keys of the input Dict, and containing a unique set
        of values among the possible combinations given by the input Dict
    '''
    keys = []
    values = []
    for key in grid:
        keys.append(key)
        values.append(grid[key])  
    list_of_permutations = [list(x) for x in np.array(np.meshgrid(*values)).T.reshape(-1,len(values))]
    
    list_of_combos = []
    for perms in list_of_permutations:
        adict = {}
        for i, value in enumerate(perms):
            adict[keys[i]] = value
        list_of_combos.append(adict)
    
    return list_of_combos

def stringify_dict(adict: dict) -> str:
    alist = []
    keys = list(sorted(list(adict.keys()))) # sort the keys so they're always in the same order!
    for key in keys:
        value = adict[key]
        if type(value) == List:
            value = ','.join(str(value))
        alist.append('%s=%s' % (str(key), str(value)))
    string = '-'.join(alist)
    return string

def unstringify_dict(string: str) -> str:
    adict = {}
    key_value_tuples = list(map(lambda x: x.split('='), string.split('-')))
    for key, value in key_value_tuples:
        adict[key] = value
    return adict
    

def update_table(dict_or_string: Union[Dict, str], hash: str) -> None:
    if os.path.exists(DICT_HASH_FILEPATH):
        with open(DICT_HASH_FILEPATH, 'r') as f:
            table = json.load(f)
    else:
        table = {'to_hash': {}, 'from_hash': {}}
    
    if type(dict_or_string) == dict:
        string = stringify_dict(dict_or_string)
    else:
        string = dict_or_string
    
    while hash in table['from_hash'] and string != table['from_hash'][hash]: # collision!
        hash += '&' # just change the hash a bit

    table['to_hash'][string] = hash
    table['from_hash'][hash] = string

    with open(DICT_HASH_FILEPATH, 'w+') as f:
        json.dump(table, f, indent=2)


def hash_fn(dict_or_string: Union[Dict, str], update: bool = False) -> str:
    
    if type(dict_or_string) == dict:
        string = stringify_dict(dict_or_string)
    else:
        string = dict_or_string
    
    hash = hashlib.md5(eval("b'%s'" % (string))).hexdigest()

    if update:
        update_table(string, hash)

    return hash

def inv_hash_fn(hash: str, unstringify: bool = False) -> Union[str, Tuple[str, Dict]]:
    with open(DICT_HASH_FILEPATH, 'r') as f:
        table = json.load(f)
    
    string = table['from_hash'][hash]
    
    if unstringify:
        return string, unstringify_dict(string)
    else:
        return string
