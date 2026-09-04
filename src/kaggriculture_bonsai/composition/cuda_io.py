from __future__ import annotations
import json,struct
from pathlib import Path
from typing import Iterable,List,Sequence

from kaggriculture_bonsai.runtime import bonsai_kg_v4 as b
from kaggriculture_bonsai.composition.module_codec import ModuleGenome

COND=struct.Struct('<HBBhHi');RULE_HEAD=struct.Struct('<BBH');GEN_HEAD=struct.Struct('<HBB');FILE_HEAD=struct.Struct('<8sII');RESULT_HEAD=struct.Struct('<8sIII');RESULT=struct.Struct('<iiHH');FRAGMENT_RESULT_HEAD=struct.Struct('<8sIIIII')
assert COND.size==12
ROUTE_RULE_SIZE=RULE_HEAD.size+4*COND.size
GENOME_SIZE=GEN_HEAD.size+8*ROUTE_RULE_SIZE
assert ROUTE_RULE_SIZE==52 and GENOME_SIZE==420

def pack_genomes(genomes:Sequence[ModuleGenome],path:str|Path)->None:
 out=bytearray(FILE_HEAD.pack(b'B4GM1\0\0\0',1,len(genomes)))
 for g in genomes:
  if len(g.rules)>8:raise ValueError('too many rules')
  body=bytearray(GEN_HEAD.pack(g.prefix_step,len(g.rules),0))
  for ri in range(8):
   if ri<len(g.rules):
    r=g.rules[ri];body+=RULE_HEAD.pack(len(r.conditions),0,r.route)
    for ci in range(4):
     if ci<len(r.conditions):
      c=r.conditions[ci];c.validate();body+=COND.pack(b.FEATURES[c.feature],b.CMP[c.cmp],0,c.index,0,c.value)
     else:body+=b'\0'*COND.size
   else:body+=b'\0'*ROUTE_RULE_SIZE
  if len(body)!=GENOME_SIZE:raise AssertionError(len(body))
  out+=body
 Path(path).write_bytes(out)

def load_results(path:str|Path):
 raw=Path(path).read_bytes();magic,version,ng,nc=RESULT_HEAD.unpack_from(raw,0)
 if magic!=b'B4RS1\0\0\0' or version!=1:raise ValueError('bad result header')
 off=RESULT_HEAD.size;expected=off+ng*nc*RESULT.size
 if len(raw)!=expected:raise ValueError(f'result size mismatch {len(raw)} != {expected}')
 rows=[]
 for g in range(ng):
  one=[]
  for c in range(nc):one.append(RESULT.unpack_from(raw,off));off+=RESULT.size
  rows.append(one)
 return rows


def load_fragment_results(path:str|Path):
 raw=Path(path).read_bytes()
 if len(raw)<FRAGMENT_RESULT_HEAD.size:raise ValueError('short fragment result header')
 magic,version,ncand,nc,start_index,total_candidates=FRAGMENT_RESULT_HEAD.unpack_from(raw,0)
 if magic!=b'B4FR1\0\0\0' or version!=1:raise ValueError('bad fragment result header')
 off=FRAGMENT_RESULT_HEAD.size;expected=off+ncand*nc*RESULT.size
 if len(raw)!=expected:raise ValueError(f'fragment result size mismatch {len(raw)} != {expected}')
 rows=[]
 for _ in range(ncand):
  one=[]
  for _ in range(nc):one.append(RESULT.unpack_from(raw,off));off+=RESULT.size
  rows.append(one)
 return {'candidates':ncand,'cases':nc,'start_index':start_index,'total_candidates':total_candidates},rows
