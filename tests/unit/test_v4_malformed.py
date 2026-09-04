from __future__ import annotations
import copy, hashlib, struct, subprocess, tempfile, unittest
from pathlib import Path
from kaggriculture_bonsai.runtime import bonsai_kg_v4 as b


def base_policy():
    return b.compile_policy({"schema_version":4,"enums":[{"name":"Mode","values":["A","B"]}],"blackboard":[{"name":"mode","type":"Mode","initial":"A"},{"name":"x","type":"I32","initial":0}],"tree":{"type":"SwitchEnum","slot":"mode","cases":{"A":{"type":"WaitTurns","turns":2},"B":{"type":"EmitMarketIntent","action":"HIRE","authority":"DOMAIN","priority":7}},"default":{"type":"Failure"}}})


def raw_invalid(cp):
    digest=hashlib.sha256(b.semantic_payload_bytes(cp)).digest()
    return b._compiled_image(cp,digest)


class MalformedV4(unittest.TestCase):
    validator=None
    @classmethod
    def setUpClass(cls):
        cls.validator=getattr(cls,"validator",None)

    def reject(self,raw,label):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/(label+".bkgc");p.write_bytes(raw)
            with self.assertRaises(Exception,msg=label):b.load_compiled(p)
            if self.validator:
                rc=subprocess.run([self.validator,str(p)],capture_output=True,text=True).returncode
                self.assertNotEqual(rc,0,label)

    def test_wrong_magic(self):
        raw=bytearray(raw_invalid(base_policy()));raw[:4]=b"NOPE";self.reject(bytes(raw),"magic")
    def test_wrong_abi(self):
        raw=bytearray(raw_invalid(base_policy()));struct.pack_into("<H",raw,4,3);self.reject(bytes(raw),"abi")
    def test_wrong_registry(self):
        raw=bytearray(raw_invalid(base_policy()));struct.pack_into("<H",raw,62,2);self.reject(bytes(raw),"registry")
    def test_count_overflow(self):
        raw=bytearray(raw_invalid(base_policy()));struct.pack_into("<H",raw,40,65535);self.reject(bytes(raw),"node_overflow")
    def test_trailing_bytes(self):self.reject(raw_invalid(base_policy())+b"x","trailing")
    def test_hash_corruption(self):
        raw=bytearray(raw_invalid(base_policy()));raw[8]^=0x55;self.reject(bytes(raw),"hash")
    def test_max_live_under_report_with_valid_hash(self):
        cp=base_policy();cp.max_live_intents=0;self.reject(raw_invalid(cp),"live_proof")
    def test_max_depth_under_report_with_valid_hash(self):
        cp=base_policy();cp.max_depth=1;self.reject(raw_invalid(cp),"depth_proof")
    def test_budget_under_report_with_valid_hash(self):
        cp=base_policy();cp.instruction_budget=1;self.reject(raw_invalid(cp),"visit_proof")
    def test_state_elision_metadata_tamper_valid_hash(self):
        cp=base_policy();cp.nodes[0].state_slot=b.NONE_U16;self.reject(raw_invalid(cp),"state_proof")
    def test_subtree_end_ffff_valid_hash(self):
        cp=base_policy();cp.nodes[0].subtree_end=b.NONE_U16;self.reject(raw_invalid(cp),"subtree_ffff")
    def test_child_cycle_valid_hash(self):
        cp=base_policy();cp.child_refs[0]=0;self.reject(raw_invalid(cp),"child_cycle")
    def test_invalid_switch_mapping_valid_hash(self):
        cp=base_policy();cp.nodes[0].p2+=1;self.reject(raw_invalid(cp),"switch")
    def test_invalid_authority_valid_hash(self):
        cp=base_policy();emit=next(n for n in cp.nodes if n.type==b.NODE_TYPES["EmitMarketIntent"]);emit.aux=(250<<8)|7;self.reject(raw_invalid(cp),"authority")
    def test_value_ref_corruption_valid_hash(self):
        cp=base_policy();cp.values[0]=b.ValueRef(b.VALUE_KINDS["BB"],b.BB_TYPES["I32"],65535,0,0);self.reject(raw_invalid(cp),"value")
    def test_predicate_padding_nonzero(self):
        cp=b.compile_policy({"schema_version":4,"blackboard":[{"name":"x","type":"I32","initial":0}],"tree":{"type":"Success","lifecycle":{"failure_if":{"lhs":{"feature":"STEP"},"op":"LT","rhs":4}}}});raw=bytearray(raw_invalid(cp));off=b.HEADER.size+len(cp.nodes)*b.NODE_STRUCT.size+len(cp.child_refs)*2+len(cp.bb)*b.BB_STRUCT.size+len(cp.values)*b.VALUE_STRUCT.size+len(cp.feature_args)*b.FEATURE_ARG_STRUCT.size;raw[off+5]=1;self.reject(bytes(raw),"predicate_padding")


def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument("--validator");args,rest=ap.parse_known_args();MalformedV4.validator=args.validator;unittest.main(argv=[sys.argv[0],*rest],verbosity=2)
if __name__=="__main__":main()
