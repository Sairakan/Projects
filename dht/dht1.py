#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 11:53:44 2018

@author: Jason Teng
"""

import xxhash
import socket
import base64
import random
import struct
import time
import select
from multiprocessing import Pool

bootstrap = ("172.18.0.252", 1337, 0xffff)
a = 3
K = 4
myID = xxhash.xxh64('netsec12').intdigest() & 0xffff
#myaddr = ('172.18.0.112', 1337)
myaddr = ('127.0.0.1', 1337)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(myaddr)

# mapping of (key: val, srcID, time)
data = {}
# mapping of IDs to number of entries, to combat DOS via multiple stores
srcIDs = {}
srcLimit = 50

pool = Pool(processes = a)

# IPv4 entries
entryformat = 'BIHIH'
entrysize = 13
timeout = 5 # 5 second timeout on messages

class routTree:
    """
    the left and right are either None (if this is a leaf node) or other
        routTrees, where the left corresponds to having
        a prefix of self.prefix + 0, and the right having a prefix
        of self.prefix + 1.
    """
    """
    the prefix is the prefix associated with this tree vertex
    l is the length in bits of the prefix
    the mask is used to determine branching in lookups/insertions
    """
    def __init__(self, p, l):
        self.prefix = p
        self.length = l
        self.nodes = []
        self.mask = 2**(15 - l)
        self.lastlookup = time.time()
    
    # n is a (address, port, id) triple
    def addNode(self, n):
        i = n[2]
        # not a leaf node
        if self.nodes == None:
            # next bit after prefix is 1; branch right
            if i & self.mask:
                self.right.addNode(n)
            # next bit after prefix is 0; branch left
            else:
                self.left.addNode(n)
        else:
            # update last lookup time
            self.lastlookup = time.time()
            # check for reinsertion
            if n in self.nodes:
                return
            # bucket not full yet; just add new node
            if len(self.nodes) < K:
                self.nodes.append(n)
            else:
                # if oldest node is still alive, reinsert at tail
                if check(self.nodes[0]):
                    self.nodes.append(self.nodes.pop(0))
                # otherwise, discard oldest node and insert new node
                else:
                    self.nodes.pop(0)
                    self.nodes.append(n)
                    return
                # check if need to split
                if self.contains(myID):
                    self.split()
                    self.addNode(n)
                # no split required, discard node
                else:
                    return
    
    # checks if this Tree's prefix matches the given node id
    def containsID(self, i):
        return self.prefix == (i >> 16-self.length)
    
    
    def split(self):
        self.left = routTree(self.prefix << 1, self.length + 1)
        self.right = routTree((self.prefix << 1) + 1, self.length + 1)
        # go through nodes and add them to left or right subtrees
        for x in self.nodes:
            if self.left.containsID(x[2]):
                self.left.addNode(x)
            else:
                self.right.addNode(x)
        # set nodes to None to indicate a non-leaf Tree
        self.nodes = None
    
    
    # returns an array of up to k entries which are closest to i
    def findNode(self, i):
        print("my nodes: " + str(self.nodes))
        print("finding nodes")
        # not a leaf node
        if self.nodes == None:
            print("non-leaf")
            # next bit after prefix is 1; branch right
            if i & self.mask:
                ans = self.right.findNode(i)
                extra = self.left.findNode(i)
            # branch left
            else:
                ans = self.left.findNode(i)
                extra = self.right.findNode(i)
            # try to add extra nodes until up to k entries in ans
            while len(extra) > 0 and len(ans) < K:
                ans.append(extra.pop(0))
            return ans
        # leaf node; just return the known nodes
        else:
            print("leaf")
            self.lastlookup = time.time()
            return self.nodes
    
    # checks for stale buckets and refreshes them
    def refresh(self):
        if self.nodes == None:
            self.right.refresh()
            self.left.refresh()
        else:
            if time.time() > self.lastlookup + 3600: # over 1 hour since last lookup
                self.findNode(self.nodes[0])
                    

# initialize routing tree with prefix 0 of length 0
rt = routTree(0, 0)

# generates a random 2-byte nonce
def generateNonce():
    return struct.unpack('>H', bytearray(random.getrandbits(8) for i in range(2)))[0]

# checks whether a given node is still alive
# n is a (address, port, id) triple
def check(n):
    addr = (n[0], n[1])
    nonce = send_ping(addr)
    t0 = time.time()
    while time.time() < t0+timeout:
        s = select.select([sock], [], [], timeout)
        if s[0]:
            reply, address = sock.recvfrom(1024)
            try:
                m = struct.unpack_from('>BIHIH', reply)
                mtype = m[0]
                mnonce = m[4]
            except:
                reply_error(address, 0, 'improper message format')
            if mtype == 1 and mnonce == nonce:
                return True
            else:
                continue
    return False
        

def send_ping(address):
    nonce = generateNonce()
    m = struct.pack('>BIHIH', 0, 2, myID, 2, nonce)
    sock.sendto(m, address)
    return nonce

def reply_ping(address, nonce):
    m = struct.pack('>BIHIH', 1, 2, myID, 2, nonce)
    sock.sendto(m, address)

def send_store(address, data):
    nonce = generateNonce()
    m = struct.pack('>BIHIHI', 2, 2, myID, 2, nonce, len(data)) + data
    sock.sendto(m, address)
    return nonce
        
def reply_store(address, nonce):
    m = struct.pack('>BIHIH', 3, 2, myID, 2, nonce)
    sock.sendto(m, address, nonce)

def send_find_node(address, nodeid):
    nonce = generateNonce()
    m = struct.pack('>BIHIHIH', 4, 2, myID, 2, nonce, 2, nodeid)
    sock.sendto(m, address)
    return nonce

def reply_find_node(address, nonce, entries):
    m = struct.pack('>BIHIHI', 5, 2, myID, 2, nonce, len(entries))
    m += entries
    sock.sendto(m, address)

def send_find_value(address, key):
    nonce = generateNonce()
    m = struct.pack('>BIHIHIH', 6, 2, myID, 2, nonce, 2, key)
    sock.sendto(m, address)
    return nonce
    
def reply_find_value(address, nonce, entries, val):
    if len(entries) == 0:
        m = struct.pack('>BIHIHII', 7, 2, myID, 2, nonce, 0, len(val))
        m += val
    else:
        m = struct.pack('>BIHIHI', 7, 2, myID, 2, nonce, len(entries))
        m += entries
        m += struct.pack('>I', 0)
    sock.sendto(m, address)

def reply_error(address, nonce, error):
    m = struct.pack('>BIHIHI', 8, 2, myID, 2, nonce, len(error)) + error
    sock.sendto(m, address)
    
    
    
    
    

# attempts to receive a reply_find_node associated with the given nonce
def get_reply_find_node(nonce, sender):
    t0 = time.time()
    while time.time() < t0+timeout:
        s = select.select([sock], [], [], 1)
        if s[0]:
            reply, address = sock.recvfrom(1024)
            try:
                m = struct.unpack_from('>BIHIH', reply)
                mtype = m[0]
                msender = m[2]
                mnonce = m[4]
            except:
                reply_error(address, 0, 'improper message format')
            if mtype == 7 and mnonce == nonce:
                elen = struct.unpack_from('>I', m, 13)[0]
                entries = []
                for i in range(elen/entrysize):
                    e = struct.unpack_from(entryformat, m, 17 + i*entrysize)
                    entry = (socket.inet_ntoa(e[1]), e[2], e[4])
                    entries.append(entry)
                return [entries, msender]
            else:
                continue
    return sender
    

# returns up to the k closest nodes to the given key
def findkclosest(key):
    closest = sorted(rt.findNode(key), key=lambda x: x[2] ^ key)
    print(closest)
    dead = []
    responded = []
    pinged = []
    while True:
        entries = closest[:a]
        nonces = [(send_find_node((e[0], e[1]), key),e) for e in entries]
        pinged.extend(entries)
        responses = [pool.apply_async(get_reply_find_node, (n,s)) for (n,s) in nonces]
        for r in responses:
            res = r.get(timeout=5)
            # got a response
            if isinstance(res,list):
                responded.append(res[1])
                # attempt to add all nodes found
                for e in res[0]:
                    rt.addNode(e)
                    if closest.count(e) == 0:
                        closest.append(e)
            else:
                dead.append(res)
                closest.remove(res)
                pinged.remove(res)
        closest.sort(key=lambda x: x[2] ^ key)
        pinged.sort(key=lambda x: x[2] ^ key)
        # check if up to the k closest have been heard from
        if pinged[:K] == closest[:K]:
            if all([responded.count(n) > 0 for n in pinged[:K]]):
                return pinged[:K]
    
                
            

# hashes the val to obtain its key and stores (key: val, srcID, currenttime) locally
def store(val, srcID):
    # if srcID not seen yet, make new entry in srcIDs
    if srcID not in srcIDs:
        srcIDs[srcID] = []
    key = xxhash.xxh64(val).intdigest() & 0xffff
    # remove expired keys in srcID's list
    srcIDs[srcID] = [k for k in srcIDs[srcID] if data[k][2] + 300 >= time.time()]
    # if over the limit, discard the store
    if len(srcIDs[srcID]) >= srcLimit:
        return
    # if key not yet stored
    if key not in data:
        data[key] = (val, srcID, time.time())
        srcIDs[srcID].append(key)
    # check for key expiration
    elif data[key][2] + 300 < time.time():
        srcIDs[data[key][1]].remove(key)
        data[key] = (val, srcID, time.time())
        srcIDs[srcID].append(key)

def publish(val):
    key = xxhash.xxh64(val).intdigest() & 0xffff
    nodes = findkclosest(key)
    for n in nodes:
        send_store((n[0],n[1]),val)

def getRawEntries(key):
    entries = rt.findNode(key)
    rawentries = ''
    # encode entries to raw bytes
    for e in entries:
        addr = struct.unpack('>I', socket.inet_aton(e[0]))[0]
        rawentries += struct.pack(entryformat, 4, addr, e[1], 2, e[2])
    return rawentries

def handleMessage(message, address):
    try:
        m = struct.unpack_from('>BIHIH', message)
        mtype = m[0]
        msender = m[2]
        mnonce = m[4]
        rt.addNode((address[0], address[1], msender))
        if mtype == 0:
            reply_ping(address, mnonce)
        elif mtype == 1:
            # reply_ping
            1
        elif mtype == 2:
            # store
            datalen = struct.unpack_from('>I', m, 13)[0]
            val = m[-datalen:]
            store(val)
            reply_store(address, mnonce)
        elif mtype == 3:
            # reply_store
            1
        elif mtype == 4:
            # find_node
            keylen = struct.unpack_from('>I', m, 13)[0]
            key = struct.unpack_from('>H', m[-keylen:])
            entries = getRawEntries(key)
            reply_find_node(address, mnonce, entries)
        elif mtype == 5:
            # reply_find_node
            elen = struct.unpack_from('>I', m, 13)[0]
            entries = []
            for i in range(elen/entrysize):
                e = struct.unpack_from(entryformat, m, 17 + i*entrysize)
                entry = (socket.inet_ntoa(e[1]), e[2], e[4])
                entries.append(entry)
            for e in entries:
                rt.addNode(e)
            
        elif mtype == 6:
            # find_value
            keylen = struct.unpack_from('>I', m, 13)[0]
            key = struct.unpack_from('>H', m[-keylen:])
            # data is stored on this node
            if data[key]:
                val = data[key][0]
                data[key] = (val, time.time()) # refresh time stamp
                reply_find_value(address, mnonce, '', val)
            # data not on this node; return known entries
            else:
                entries = getRawEntries(key)
                reply_find_value(address, mnonce, entries, '')
        elif mtype == 7:
            # reply_find_value
            1
        elif mtype == 8:
            # error
            1
    except:
        reply_error(address, 0, 'improper message format')
    
def run():
    f = open('dht_data', 'r')
    mydata = []
    for line in f:
        # store a data entry with the current time
        mydata.append(base64.b64decode(line))
    for d in mydata:
        store(d, myID)
    rt.addNode(bootstrap)
    #send_find_node((bootstrap[0],bootstrap[1]), myID)
    for d in mydata:
        publish(d)
    checktime = time.time()
    republishtime = time.time()
    while True:
        s = select.select([sock], [], [], 1)
        if s[0]:
            message, address = sock.recvfrom(1024)
            handleMessage(message, address)
            # check over k-buckets every 30 seconds
            if time.time() > checktime + 30:
                rt.refresh()
                checktime = time.time()
            # republish data every 5 minutes to combat the 5 minute cache timeout
            if time.time() > republishtime + 300:
                for d in mydata:
                    publish(d)
            
            
run()




















