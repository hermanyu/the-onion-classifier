# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 09:58:54 2022

@author: herman
"""

import requests
import time

def get_reddit(content = 'submission', params = {'subreddit':'datascience', 'fields':('title', 'selftext','created_utc'), 'size':'100'}, verbose=False):
    url = 'https://api.pushshift.io/reddit/search/'
    post_endpt = 'submission/'
    comment_endpt = 'comment/'
    
    if content == 'submission':
        url = url+post_endpt
        
    elif content == 'comment':
        url = url+comment_endpt
        
    else:
        print('"content" parameter value unrecognized. Valid parameter values are "submission" and "comment".')
        return None
    
    req = requests.get(url, params=params)
    
    if verbose == True:
        print('Passed URL: {}'.format(req.url))
        print('Status Code: {}'.format(req.status_code))
    
    return req.json()['data']
    

def mass_collect_reddit(content = 'submission', params = {'subreddit':'datascience', 'fields':('title', 'selftext', 'created_utc'), 'size':'100'},iters=1, verbose=False, delay=True):
    json = []
    epoch = 0
    new_params = params
    for i in range(iters):
        json = json+get_reddit(content,new_params, verbose=verbose)
        epoch = json[-1]['created_utc']
        new_params['before'] = epoch
        
        if verbose == True:
            print(f'iteration {i}')
            print(epoch)
        
        if delay == True:
            time.sleep(0.8) # add a time delay in the loop to avoid DDoS-ing the server accidentally
            
    return json
    
    
    
