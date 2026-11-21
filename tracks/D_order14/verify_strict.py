"""Run src/verify_sos_standalone.verify with every `assert` rewritten to an explicit raise.
The original file is not modified; its logic is unchanged. Safe under python -O.
Usage: python -O verify_strict.py <certificate.json> <output.json>"""
import ast,sys,json,os
ROOT=os.path.expanduser('~/gcm-logconcave-counterexamples');src=open(ROOT+'/src/verify_sos_standalone.py').read()
class NoAssert(ast.NodeTransformer):
    def visit_Assert(self,n):
        msg=n.msg if n.msg is not None else ast.Constant(f'check failed at line {n.lineno}')
        exc=ast.Call(func=ast.Name('RuntimeError',ast.Load()),args=[msg],keywords=[])
        return ast.copy_location(ast.If(test=ast.UnaryOp(ast.Not(),n.test),body=[ast.Raise(exc=exc,cause=None)],orelse=[]),n)
tree=ast.fix_missing_locations(NoAssert().visit(ast.parse(src)))
if any(isinstance(n,ast.Assert) for n in ast.walk(tree)):raise SystemExit('assert survived rewrite')
ns={'__name__':'verify_strict_module','__file__':ROOT+'/src/verify_sos_standalone.py'};os.chdir(ROOT)
exec(compile(tree,'verify_sos_standalone[strict]','exec'),ns)
r=ns['verify'](sys.argv[1]);r.update(certificate=sys.argv[1],python_optimize_flag=sys.flags.optimize,assert_rewritten=True)
if not (r.get('exact_identity') and r.get('positive_definite_grams') and r.get('ibp_identities_verified')):raise RuntimeError('verifier did not confirm all checks')
json.dump(r,open(sys.argv[2],'w'),indent=2);print(json.dumps(r))
